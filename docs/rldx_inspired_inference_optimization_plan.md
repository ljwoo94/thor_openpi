# RLDX-Inspired Inference Optimization Plan for OpenPI pi0.5

This document explains how to adapt the RLDX-1 inference optimization ideas to
the OpenPI pi0.5 PyTorch path in this repository. It focuses on the current
state where `torch.compile(self.sample_actions, mode="max-autotune")` already
produces separate Nsight Systems regions for:

- vision tower / image embedding
- VLM prefix forward / KV prefill
- each denoising step

The goal is to decide how far we can push toward an RLDX-style GraphSafe VLA
and fullgraph execution.

## Current pi0.5 Inference Path

Main files:

- Server: [`scripts/serve_policy.py`](../scripts/serve_policy.py)
- WebSocket server: [`src/openpi/serving/websocket_policy_server.py`](../src/openpi/serving/websocket_policy_server.py)
- Policy wrapper: [`src/openpi/policies/policy.py`](../src/openpi/policies/policy.py)
- PyTorch model: [`src/openpi/models_pytorch/pi0_pytorch.py`](../src/openpi/models_pytorch/pi0_pytorch.py)
- PaliGemma/Gemma wrapper: [`src/openpi/models_pytorch/gemma_pytorch.py`](../src/openpi/models_pytorch/gemma_pytorch.py)

High-level runtime:

```text
WebSocket request
  -> Policy.infer
  -> input transforms / normalization
  -> Observation.from_dict
  -> PI0Pytorch.sample_actions
      -> preprocess observation
      -> embed_prefix(images, language)
      -> PaliGemma prefix forward with use_cache=True
      -> iterative denoising loop
          -> embed_suffix(state, x_t, timestep)
          -> Gemma expert forward with prefix KV cache
          -> action_out_proj
          -> Euler update
  -> output transforms / unnormalize
  -> WebSocket response
```

The current PyTorch model already compiles `sample_actions`:

```python
torch.set_float32_matmul_precision("high")
self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
```

Location: [`src/openpi/models_pytorch/pi0_pytorch.py`](../src/openpi/models_pytorch/pi0_pytorch.py)

## Why Current Compile Splits in Nsight

Seeing separate compiled/graph regions for vision tower, VLM forward, and each
denoising step is expected. The current `sample_actions` function is not a
single static graph in the RLDX sense.

Likely split causes:

1. **Different computation phases**
   Prefix prefill and suffix denoising call different submodels and different
   cache modes.

2. **Dynamic cache object**
   Prefix forward uses:

   ```python
   use_cache=True
   past_key_values=None
   ```

   and later passes `past_key_values` into the expert forward. HuggingFace cache
   classes are usually difficult for `fullgraph=True` and CUDA graph replay
   unless converted to static tensor buffers.

3. **Python control flow**
   `sample_actions` uses a Python `while time >= -dt / 2` loop. Even if Dynamo
   captures parts of it, it commonly specializes each iteration or emits
   separate regions.

4. **Forward-time tensor construction**
   The hot path constructs timesteps, masks, position ids, and attention masks
   each call.

5. **Forward-time config mutation**
   The current inference path sets attention implementation inside the hot path:

   ```python
   self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
   self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
   ```

   These should be moved out of `sample_actions` / `denoise_step` before trying
   fullgraph capture.

6. **Preprocessing and Observation object boundary**
   `sample_actions` still calls `_preprocess_observation`, which consumes an
   `Observation` object and dictionaries. This is convenient for normal
   inference but not ideal for fullgraph capture.

## Can We Use RLDX GraphSafeVLA Directly?

No. RLDX `GraphSafeVLA` cannot be imported and applied directly to pi0.5.

RLDX GraphSafe modules assume:

- Qwen3-VL-derived RLDX backbone
- MSAT action model
- RLDX cognition-token interface
- RLDX action encoder/decoder modules
- RLDX memory buffer layout
- custom Triton kernels for RLDX block shapes

pi0.5 uses:

- PaliGemma vision/language prefix
- Gemma action expert
- HF-style KV cache
- flow matching over action chunks
- AdaRMS conditioning for pi0.5

Therefore, what can be imported from RLDX is the **optimization architecture**,
not the modules:

- static graph wrapper pattern
- preallocated input/output buffers
- first-call shape capture
- eager fallback on shape drift
- compile-mode dispatch
- RTC prefix-inpainting idea
- correctness comparison harness
- benchmark structure

## Practical Target: GraphSafePi05, Not GraphSafeVLA

The correct adaptation is to build a pi0.5-specific graph-safe wrapper:

```text
GraphSafePi05
  -> GraphSafePrefixPrefill    optional / later
  -> GraphSafeDenoiseLoop      first target
  -> static buffers
  -> optional CUDA graph replay
  -> optional torch.compile(fullgraph=True)
```

Do not start with the full VLA. Start with the denoising loop because it is the
most repeated and most shape-stable part.

## Recommended Optimization Stages

### Stage 0: Keep Eager Baseline

Add a serving option that disables compile. Current compile is unconditional.
We need a reliable baseline for correctness and profiling.

Recommended CLI/config shape:

```text
--compile none
--compile sample-actions
--compile denoise-step
--compile graphsafe-denoise
--compile graphsafe-full
```

Expected change location:

- [`scripts/serve_policy.py`](../scripts/serve_policy.py)
- [`src/openpi/policies/policy_config.py`](../src/openpi/policies/policy_config.py)
- [`src/openpi/models_pytorch/pi0_pytorch.py`](../src/openpi/models_pytorch/pi0_pytorch.py)

### Stage 1: Hoist Hot-Path Mutations

Move these out of inference:

```python
self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
```

Set them once in model init or policy load.

Impact:

- fewer graph breaks
- easier fullgraph attempts
- clearer inference semantics

### Stage 2: Split `sample_actions`

Refactor into explicit phases:

```python
def prefill_prefix(self, observation):
    images, img_masks, lang_tokens, lang_masks, state = ...
    prefix_embs, prefix_pad_masks, prefix_att_masks = ...
    past_key_values = ...
    return state, prefix_pad_masks, past_key_values

def denoise_loop(self, state, prefix_pad_masks, past_key_values, noise, num_steps):
    x_t = noise
    for step in range(num_steps):
        x_t = self.denoise_step(...)
    return x_t
```

This lets us compile or graph-capture only `denoise_loop` first.

Impact:

- isolates repeated work
- easier correctness checks
- makes future static buffers natural

### Stage 3: GraphSafe Denoising Loop

Create a new module, for example:

```text
src/openpi/models_pytorch/inference/
  graph_safe_pi05.py
  compile_dispatch.py
  cuda_graph.py
```

Initial wrapper:

```python
class GraphSafePi05Denoise(torch.nn.Module):
    def __init__(self, model, *, action_horizon, action_dim, num_steps, device):
        super().__init__()
        self.model = model
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.dt = -1.0 / num_steps
        self.register_buffer(
            "timesteps",
            torch.arange(num_steps, device=device, dtype=torch.float32) / num_steps,
            persistent=False,
        )

    def forward(self, state, prefix_pad_masks, past_key_values, init_noise):
        x_t = init_noise
        bsize = state.shape[0]
        for i in range(self.num_steps):
            # pi0 convention: time starts at 1 and decreases to 0
            t = 1.0 - i / self.num_steps
            timestep = torch.full((bsize,), t, device=state.device, dtype=torch.float32)
            v_t = self.model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep,
            )
            x_t = x_t + self.dt * v_t
        return x_t
```

Then test:

```python
compiled = torch.compile(wrapper, mode="max-autotune", fullgraph=True)
```

Expected blocker:

- `past_key_values` object may not be fullgraph-compatible.

If blocked, move to Stage 4.

### Stage 4: Static KV Cache

This is likely the central requirement for RLDX-style fullgraph.

Current pi0.5 prefix prefill returns HF cache objects. RLDX avoids this class of
problem by turning model state into static tensors and buffers.

For pi0.5, investigate using or adapting:

- `transformers.cache_utils.StaticCache`
- patched PaliGemma/Gemma static-cache support in
  [`src/openpi/models_pytorch/transformers_replace`](../src/openpi/models_pytorch/transformers_replace)

Goal:

```text
prefix prefill writes K/V tensors into fixed buffers
denoise loop reads fixed K/V tensors
no cache object mutation in compiled loop
```

A graph-safe cache interface should look like:

```python
@dataclass
class StaticPrefixCache:
    key: list[torch.Tensor]
    value: list[torch.Tensor]
    prefix_len: torch.Tensor | int
```

or, better for compile:

```python
class StaticPrefixCacheBuffers(torch.nn.Module):
    def __init__(...):
        self.register_buffer("k_layer_0", ...)
        self.register_buffer("v_layer_0", ...)
        ...
```

Impact:

- required for fullgraph denoise loop
- required for CUDA graph replay
- probably not required for basic `torch.compile` partial speedups

### Stage 5: CUDA Graph Replay for Denoise Loop

Once inputs are tensors and shapes are fixed:

```python
static_state.copy_(state)
static_noise.copy_(noise)
static_prefix_cache.copy_(...)
graph.replay()
```

Adapt the RLDX pattern from:

- [`../../rldx/inference/engine/cuda_graph.py`](../../rldx/inference/engine/cuda_graph.py)
- [`../../rldx/inference/serve_optimization.py`](../../rldx/inference/serve_optimization.py)

Required properties:

- fixed batch size, likely `B=1`
- fixed `action_horizon`
- fixed `action_dim`
- fixed prompt token length or padded prompt length
- fixed prefix token length
- explicit `init_noise` buffer
- no allocation in hot replay

### Stage 6: Optional Full GraphSafe VLA

Only after graph-safe denoise works:

```text
GraphSafePi05Full
  -> image embedding
  -> language embedding
  -> prefix cache fill
  -> denoise loop
```

This is harder because image/token lengths and HF cache behavior are more
dynamic. Full VLA capture likely requires:

- fixed 3-camera image set
- fixed 224x224 image tensors
- fixed tokenized prompt length, padded to `max_token_len`
- static cache buffers
- no dict/object work inside captured forward

## Kernel Fusion Plan

Do not port RLDX kernels directly. RLDX fuses Qwen/MSAT-specific operations.
pi0.5 needs Gemma/PaliGemma-specific fusion.

Potential candidates after profiling:

1. **AdaRMSNorm + residual gate**
   Location:
   [`src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`](../src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py)

2. **Gemma MLP**
   `gate_proj`, activation, `up_proj`, multiply, `down_proj`.

3. **RoPE preparation**
   If repeated per denoise step with same positions, precompute first. Fuse only
   if still hot.

4. **Action input projection + time conditioning**
   Small but repeated `num_steps` times.

Kernel fusion should come after:

- baseline profile
- submodule compile
- graph-safe denoise loop
- static KV cache

## RTC / Chunk Optimization

OpenPI already has client-side chunk amortization:

- [`packages/openpi-client/src/openpi_client/action_chunk_broker.py`](../packages/openpi-client/src/openpi_client/action_chunk_broker.py)

This returns one action at a time from a cached chunk and only calls the policy
when the chunk is exhausted.

RLDX's next step is smoother chunk overlap:

- store previous predicted chunk
- inject prefix from previous chunk into new chunk
- optionally train with prefix conditioning

For pi0.5, implement this as a separate policy wrapper before changing model
training:

```text
ActionChunkBroker
  -> OverlapChunkBroker
      stores previous chunk
      sends action_prefix / prefix_len to server
```

Then, if fine-tuning is possible, add trained RTC to the flow-matching training
objective.

## Recommended Immediate Tasks

1. Add compile-mode flag and preserve eager baseline.
2. Add timing instrumentation around:
   - transforms
   - prefix embedding
   - prefix forward/cache fill
   - denoise loop
   - each denoise step
   - output transforms
3. Move attention implementation config mutation out of hot path.
4. Refactor `sample_actions` into `prefill_prefix` and `denoise_loop`.
5. Try compiling only `denoise_step` and only `denoise_loop`.
6. Prototype `GraphSafePi05Denoise` with explicit `init_noise`.
7. Investigate static cache compatibility.
8. Attempt CUDA graph replay for denoise loop.
9. Only then explore full VLA graph and custom kernels.

## Expected Feasibility

| Optimization | Feasibility | Risk | Expected Impact |
|---|---:|---:|---:|
| Timing instrumentation | High | Low | High diagnostic value |
| Compile-mode flag | High | Low | Medium |
| Hoist config mutations | High | Low | Medium |
| Split prefill/denoise | High | Medium | High enabling value |
| Compile `denoise_step` | High | Low/medium | Medium |
| Compile denoise loop fullgraph | Medium | Medium | Medium/high |
| Static KV cache | Medium | High | High enabling value |
| CUDA graph denoise loop | Medium | Medium/high | High |
| Full VLA CUDA graph | Low/medium | High | High if successful |
| Custom Triton fusion | Low | High | Unknown until profiling |

## Main Answer

In the current circumstance, we should not try to wrap the existing
`sample_actions` directly as one RLDX-style fullgraph VLA. The Nsight split
shows that Dynamo/Inductor is already breaking the computation into natural
phases. To reach RLDX-style fullgraph, we first need to make pi0.5 graph-safe:

1. remove hot-path mutations,
2. split prefix prefill from denoising,
3. turn cache objects into static tensor buffers,
4. precompute masks/timesteps/positions,
5. capture or compile the denoising loop,
6. then attempt the full VLA.

The nearest practical equivalent of RLDX `GraphSafeVLA` is a new
`GraphSafePi05Denoise` wrapper first, followed later by `GraphSafePi05Full`.
