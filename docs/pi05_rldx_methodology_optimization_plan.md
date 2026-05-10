# pi0.5 Inference Optimization Plan Using RLDX Methodology

This plan applies the RLDX inference methodology to OpenPI pi0.5, starting
from the highest expected impact with the smallest behavior-preserving changes.

The core principle is:

```text
measure -> isolate phases -> compile stable subgraphs -> graph-safe wrappers
-> CUDA graph / fullgraph -> kernel fusion
```

Do not start with custom kernels. First make the inference path measurable,
configurable, and structurally ready for graph capture.

## Priority 1: Add Timing Instrumentation

Goal: identify the real bottleneck before changing the model path.

Add timing around:

- input transforms / normalization
- `Observation.from_dict`
- image embedding / vision tower
- language embedding
- prefix VLM forward / KV cache fill
- denoising loop total
- each `denoise_step`
- output projection
- output transforms / unnormalization
- WebSocket receive/send overhead

Target files:

- [`src/openpi/policies/policy.py`](../src/openpi/policies/policy.py)
- [`src/openpi/models_pytorch/pi0_pytorch.py`](../src/openpi/models_pytorch/pi0_pytorch.py)
- [`src/openpi/serving/websocket_policy_server.py`](../src/openpi/serving/websocket_policy_server.py)

Expected impact:

- High diagnostic value.
- Low implementation risk.
- Required before deciding whether fullgraph or kernel fusion matters.

## Priority 2: Make Compile Mode Configurable

Current PyTorch pi0.5 unconditionally compiles `sample_actions`:

```python
self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
```

Replace this with an explicit serving/model option:

```text
--compile none
--compile sample-actions
--compile denoise-step
--compile denoise-loop
--compile graphsafe-denoise
```

Target files:

- [`scripts/serve_policy.py`](../scripts/serve_policy.py)
- [`src/openpi/policies/policy_config.py`](../src/openpi/policies/policy_config.py)
- [`src/openpi/models_pytorch/pi0_pytorch.py`](../src/openpi/models_pytorch/pi0_pytorch.py)

Expected impact:

- High engineering value.
- Lets us compare eager, current compile, and narrower compile strategies.
- Prevents debugging compile behavior blindly.

## Priority 3: Hoist Hot-Path Config Mutations

Move attention implementation mutation out of inference functions.

Current hot-path mutations:

```python
self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
```

These should be set once during model initialization or policy loading.

Target file:

- [`src/openpi/models_pytorch/pi0_pytorch.py`](../src/openpi/models_pytorch/pi0_pytorch.py)

Expected impact:

- Medium to high compile stability improvement.
- Very small code change.
- Reduces graph breaks and makes inference semantics cleaner.

## Priority 4: Split `sample_actions` Into Phases

Refactor `sample_actions` into explicit stages:

```python
def prefill_prefix(self, observation):
    ...
    return state, prefix_pad_masks, past_key_values

def denoise_loop(self, state, prefix_pad_masks, past_key_values, noise, num_steps):
    ...
    return actions

def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
    ...
    return velocity
```

This mirrors RLDX's separation between:

- input preparation
- cached context
- repeated action-model denoising
- output decoding

Target file:

- [`src/openpi/models_pytorch/pi0_pytorch.py`](../src/openpi/models_pytorch/pi0_pytorch.py)

Expected impact:

- High enabling value.
- Behavior-preserving if done carefully.
- Makes it possible to compile or graph-capture only the repeated denoising
  path instead of the full VLA.

## Priority 5: Compile Only `denoise_step`

After phase splitting, compile the repeated step first:

```python
self.denoise_step = torch.compile(
    self.denoise_step,
    mode="max-autotune-no-cudagraphs",
)
```

Why this before fullgraph:

- It repeats `num_steps` times.
- It is narrower than `sample_actions`.
- It avoids compiling preprocessing and prefix prefill.
- It can preserve current cache behavior.

Target file:

- [`src/openpi/models_pytorch/pi0_pytorch.py`](../src/openpi/models_pytorch/pi0_pytorch.py)

Expected impact:

- Medium.
- Lower risk than compiling all of `sample_actions`.

## Priority 6: Compile `denoise_loop`

Compile the whole denoising loop once `denoise_step` is stable:

```python
self.denoise_loop = torch.compile(
    self.denoise_loop,
    mode="max-autotune",
)
```

Potential blockers:

- `past_key_values` dynamic cache object.
- Python loop specialization.
- dynamic mask and position construction.

Expected impact:

- Medium to high if the denoising loop dominates latency.
- Medium risk.

## Priority 7: Build `GraphSafePi05Denoise`

Create a pi0.5-specific graph-safe denoising wrapper. This is the first direct
RLDX-style adaptation.

Proposed location:

```text
src/openpi/models_pytorch/inference/
  graph_safe_pi05.py
  compile_dispatch.py
  cuda_graph.py
```

Initial wrapper responsibilities:

- precompute timesteps
- store fixed `dt`
- assume fixed `action_horizon`
- assume fixed `action_dim`
- accept explicit `init_noise`
- avoid object/dict work inside `forward`
- avoid dynamic allocation inside `forward` where possible

Sketch:

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
            1.0 - torch.arange(num_steps, device=device, dtype=torch.float32) / num_steps,
            persistent=False,
        )

    def forward(self, state, prefix_pad_masks, past_key_values, init_noise):
        x_t = init_noise
        bsize = state.shape[0]
        for i in range(self.num_steps):
            timestep = self.timesteps[i].expand(bsize)
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

Expected impact:

- High enabling value.
- Required for CUDA graph and fullgraph attempts.

## Priority 8: Static KV Cache

This is likely the main blocker for RLDX-style fullgraph.

Current pi0.5 prefix prefill returns HuggingFace-style `past_key_values`.
For graph-safe replay, convert this into static tensor buffers or use
`StaticCache` if compatible.

Goal:

```text
prefix prefill writes K/V tensors into fixed buffers
denoise loop reads fixed K/V tensors
compiled loop does not mutate dynamic cache objects
```

Investigate:

- `transformers.cache_utils.StaticCache`
- patched transformer code under
  [`src/openpi/models_pytorch/transformers_replace`](../src/openpi/models_pytorch/transformers_replace)

Expected impact:

- High enabling value.
- Medium to high implementation risk.

## Priority 9: CUDA Graph Replay for Denoising

Once denoise inputs are static tensors:

```python
static_state.copy_(state)
static_noise.copy_(noise)
static_cache.copy_(cache)
graph.replay()
```

Requirements:

- fixed batch size, likely `B=1`
- fixed `action_horizon`
- fixed `action_dim`
- fixed prefix length
- explicit noise input
- no shape drift

Expected impact:

- High if kernel launch overhead is significant.
- Medium to high risk.

## Priority 10: Full GraphSafe pi0.5 VLA

Only attempt after graph-safe denoising works.

Full graph-safe target:

```text
image embedding
language embedding
prefix prefill / static KV cache
denoise loop
action output
```

Requirements:

- fixed camera set
- fixed image resolution
- fixed padded token length
- static prefix cache
- no dict/object work inside captured forward

Expected impact:

- Potentially high.
- High implementation risk.

## Priority 11: Kernel Fusion

Kernel fusion should be last and profile-driven.

Potential candidates:

- AdaRMSNorm + gate/residual
- Gemma MLP projection/activation path
- repeated RoPE preparation
- action projection + time conditioning
- expert attention epilogues

Target files:

- [`src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`](../src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py)
- future custom kernels under `src/openpi/models_pytorch/inference/kernels/`

Expected impact:

- Unknown until profiling.
- High maintenance cost.
- Do not port RLDX kernels directly; use them as design references only.

## Recommended First Sprint

Implement only these first:

1. Timing instrumentation.
2. Compile-mode flag.
3. Hoist attention config mutations.
4. Split `sample_actions` into `prefill_prefix`, `denoise_loop`, and
   `denoise_step`.

Success criteria:

- eager and compiled outputs match within tolerance
- timing report is returned in `policy_timing`
- existing server API remains compatible
- current compiled mode can still be reproduced
- denoising loop can be measured independently

## Expected Feasibility Table

| Step | Feasibility | Risk | Expected Value |
|---|---:|---:|---:|
| Timing instrumentation | High | Low | High |
| Compile-mode flag | High | Low | High |
| Hoist config mutations | High | Low | Medium/high |
| Split `sample_actions` | High | Medium | High |
| Compile `denoise_step` | High | Low/medium | Medium |
| Compile `denoise_loop` | Medium | Medium | Medium/high |
| `GraphSafePi05Denoise` | Medium | Medium | High |
| Static KV cache | Medium | High | High |
| CUDA graph denoise | Medium | High | High |
| Full GraphSafe VLA | Low/medium | High | High |
| Kernel fusion | Low | High | Unknown |

## Guiding Rule

RLDX's strongest transferable idea is not the exact code. It is the sequence:

```text
make the runtime explicit
make shapes static
make state explicit
compile the repeated core
capture stable buffers
fuse only the proven hot kernels
```

For pi0.5, the repeated core is the flow-matching denoising loop. Optimize that
before attempting full VLA graph capture.
