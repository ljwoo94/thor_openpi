# OpenPI Inference Acceleration — Full Implementation Plan

## Overview & Profiling Targets

| Component | % Latency | Jetson Thor | RTX 5090 |
|-----------|-----------|-------------|----------|
| VLM Prefix Forward | ~40% | ~76ms | ~32ms |
| Vision Tower (SigLIP × N cams) | ~20% | ~38ms | ~16ms |
| Denoising Loop (10 steps) | ~35% | ~67ms | ~28ms |
| Misc (preprocess, embed, proj) | ~5% | ~9ms | ~4ms |
| **Total** | | **~190ms** | **~80ms** |

---

## Implementation Status

| OPT | Name | Expected Savings | Status | Files |
|-----|------|-----------------|--------|-------|
| 1 | SDPA Attention | −15-25ms | ✅ Done | `gemma_pytorch.py` |
| 2 | Separate Compile Units | −10-20ms | ✅ Done | `pi0_pytorch.py` |
| 3 | While→For Loop | −5ms | ✅ Done | `pi0_pytorch.py` |
| 4 | Static KV Cache | −5-10ms | ✅ Done | `pi0_pytorch.py` |
| 5 | Batch Camera Images | −10-20ms | ☐ Pending | `pi0_pytorch.py` |
| 6 | Graph Break Elimination | −10-15ms | ✅ Done | 4 files (see below) |
| 7 | QKV + MLP Fusion | −5-10ms | ✅ Done | `fusion_utils.py`, `gemma_pytorch.py` |
| 8 | Reduce Steps (10→5) | −33ms | ☐ Pending | config change |
| 9B | Token Merging (ToMe) | −5-15ms | ✅ Done | `tome_utils.py` |
| 9A/C | Vision Resolution | −15ms | ☐ Pending | preprocessing |
| 10 | Channels-Last Memory | −3-8ms | ✅ Done | `pi0_pytorch.py` |
| 11-19 | Quantization/Sparsity/Distillation | varies | ☐ Pending | see below |

**8 of 19 optimizations implemented.** All changes include `OPT-N:` comment prefixes for individual testing.

---

## ✅ OPT-1: SDPA Attention (−15-25ms)

**File**: `src/openpi/models_pytorch/gemma_pytorch.py` — `compute_layer_complete()`, lines 211-221

**What was done**: Replaced `modeling_gemma.eager_attention_forward()` (3 separate kernels: Q×K^T → softmax → ×V, materializes full L×L matrix) with `F.scaled_dot_product_attention()` (single fused kernel, FlashAttention/memory-efficient backend).

```python
# OPT-1: SDPA attention — fused single-kernel attention (FlashAttention-style)
num_kv_groups = self.paligemma.language_model.layers[layer_idx].self_attn.num_key_value_groups
key_states_rep = modeling_gemma.repeat_kv(key_states, num_kv_groups)
value_states_rep = modeling_gemma.repeat_kv(value_states, num_kv_groups)
att_output = F.scaled_dot_product_attention(
    query_states, key_states_rep, value_states_rep,
    attn_mask=attention_mask, scale=scaling,
)
att_output = att_output.transpose(1, 2).contiguous()
```

Also fixed: hardcoded `1 * 8 * head_dim` → `num_heads * head_dim` for dynamic head count.

**Risk**: Very Low — SDPA is numerically equivalent
**Verification**: Compare max abs diff of action outputs with same noise seed

---

## ✅ OPT-2: Separate Compile Units (−10-20ms)

**File**: `src/openpi/models_pytorch/pi0_pytorch.py` — `__init__()`, lines 113-118

**What was done**: Changed compilation target from `sample_actions` (contains loop → graph breaks at boundary, no CUDA graph capture) to `denoise_step` with `fullgraph=True`.

```python
# OPT-2: Compile denoise_step separately with fullgraph=True
if config.pytorch_compile_mode is not None:
    self.denoise_step = torch.compile(
        self.denoise_step, mode=config.pytorch_compile_mode, fullgraph=True
    )
```

**Risk**: Low — requires OPT-3 and OPT-6 to eliminate all graph breaks inside denoise_step
**Verification**: Run with `TORCH_LOGS=graph_breaks` to confirm no breaks

---

## ✅ OPT-3: While→For Loop (−5ms)

**File**: `src/openpi/models_pytorch/pi0_pytorch.py` — `sample_actions()`, lines 437-454

**What was done**: Replaced `while time >= -dt/2` (tensor comparison → graph break) with `for step_idx in range(num_steps)`.

```python
# OPT-3: Replace while loop with deterministic for loop.
x_t = noise
for step_idx in range(num_steps):
    t = 1.0 - step_idx / num_steps
    expanded_time = torch.full((bsize,), t, dtype=torch.float32, device=device)
    v_t = self.denoise_step(state, prefix_pad_masks, static_kv, x_t, expanded_time)
    x_t = x_t + dt * v_t
```

**Time values**: Old t=[1.0, 0.9, ..., 0.1], New t=[1.0, 0.9, ..., 0.1] ✓ Identical.

**Risk**: None — mathematically identical

---

## ✅ OPT-4: Static KV Cache (−5-10ms)

**File**: `src/openpi/models_pytorch/pi0_pytorch.py` — `sample_actions()`, lines 414-433

**What was done**: After prefix prefill, convert DynamicCache to pre-allocated static KV buffers `[B, H, L_prefix + L_suffix, D]`. Prefix portion is copied once; suffix K/V are concatenated per step via `torch.cat` in `GemmaAttention.forward()` (line 309 of modeling_gemma.py).

```python
# OPT-4: Convert DynamicCache to static pre-allocated KV buffers.
suffix_embs_probe, suffix_pad_masks_probe, _, _ = self.embed_suffix(
    state, noise, torch.ones(bsize, dtype=torch.float32, device=device),
)
L_suffix = suffix_pad_masks_probe.shape[1]
num_layers = len(past_key_values)
static_kv = []
for layer_idx in range(num_layers):
    pk, pv = past_key_values[layer_idx]
    B, H, L_prefix, D = pk.shape
    full_k = torch.empty(B, H, L_prefix + L_suffix, D, dtype=pk.dtype, device=pk.device)
    full_v = torch.empty(B, H, L_prefix + L_suffix, D, dtype=pv.dtype, device=pv.device)
    full_k[:, :, :L_prefix, :] = pk
    full_v[:, :, :L_prefix, :] = pv
    static_kv.append((full_k, full_v))
```

**Compatibility**: The `static_kv` list-of-tuples format works with `GemmaAttention.forward()` which accesses `past_key_value[self.layer_idx][0/1]` and does `torch.cat` on the sequence dimension.

**Risk**: Low
**Note**: Full in-place write (eliminating the per-step `torch.cat`) would require modifying `modeling_gemma.py`'s attention forward — deferred to a future OPT.

---

## ☐ OPT-5: Batch Camera Images (−10-20ms)

**Target**: `pi0_pytorch.py` — `embed_prefix()`, lines 224-233

**Problem**: 4 cameras → 4 sequential SigLIP forwards. GPU underutilized at batch=1.

**Implementation** (not yet applied):
```python
# In embed_prefix, replace sequential loop:
valid_imgs = [img for img, mask in zip(images, img_masks) if mask.any()]
if valid_imgs:
    batched = torch.cat(valid_imgs, dim=0)  # [N_cams, C, H, W]
    all_embs = self.paligemma_with_expert.embed_image(batched)  # [N_cams, 256, D]
    per_cam_embs = all_embs.split(1, dim=0)  # List of [1, 256, D]
```

**Precision note**: LayerNorm and softmax in ViT are computed per-token (across feature dim), not across batch. Batched vs. sequential gives identical results up to CUDA kernel selection non-determinism (~1e-7 FP32, ~1e-3 BF16). Safe for robot control.

**Risk**: Low — must handle masked cameras correctly

---

## ✅ OPT-6: Graph Break Elimination (−10-15ms)

**Files**: 4 files, 7 locations total

All data-dependent branches and per-call attribute mutations that prevent `torch.compile` from capturing full CUDA graphs have been eliminated:

| # | File | Line(s) | Original Pattern | Fix |
|---|------|---------|-----------------|-----|
| 1 | `pi0_pytorch.py` | 120-127 | Per-call `config._attn_implementation = "eager"` | Set once at `__init__` |
| 2 | `pi0_pytorch.py` | 122-125 | Runtime `q_proj.weight.dtype == bfloat16` check | Cached as `self._uses_bfloat16` bool at init |
| 3 | `pi0_pytorch.py` | 129-133 | Runtime `state_proj.weight.dtype == float32` check | Cached as `self._state_proj_is_float32` bool at init |
| 4 | `pi0_pytorch.py` | 275 | `self.state_proj.weight.dtype == torch.float32` | Use cached `self._state_proj_is_float32` |
| 5 | `gemma_pytorch.py` | 235 | `att_output.dtype != layer...o_proj.weight.dtype` | Unconditional `.to(layer...dtype)` cast |
| 6 | `gemma_pytorch.py` | 243 | `layer.mlp.up_proj.weight.dtype == torch.bfloat16` | Unconditional `.to(layer.mlp.down_proj.weight.dtype)` |
| 7 | `modeling_gemma.py` | 505-507 | `self.layers[0]...dtype == torch.bfloat16` → conditional cast | Unconditional `.to(self.layers[0]...dtype)` |
| 8 | `modeling_siglip.py` | 776-779 | Same pattern in SigLIP vision transformer | Unconditional `.to(encoder.layers[0]...dtype)` |

**Risk**: Low — unconditional casts are always correct regardless of model precision

---

## ✅ OPT-7: QKV + MLP Fusion (−5-10ms)

**Files**:
- **NEW** `src/openpi/models_pytorch/fusion_utils.py` — post-load weight packing utilities
- `src/openpi/models_pytorch/gemma_pytorch.py` — fused forward paths in `compute_layer_complete()`

**What was done**:

1. **`fusion_utils.py`** provides 4 functions:
   - `fuse_qkv_projections(model)` — Packs Q+K+V weights into single `qkv_proj` (3 GEMMs → 1)
   - `fuse_gate_up_projections(model)` — Packs gate+up weights into single `gate_up_proj` (2 GEMMs → 1)
   - `unfuse_qkv_projections(model)` — Restores separate projections (for checkpoint saving)
   - `unfuse_gate_up_projections(model)` — Restores separate projections

2. **`gemma_pytorch.py`** detects fusion via `_qkv_fused` / `_gate_up_fused` flags:
   ```python
   # OPT-7: Use fused QKV projection if available (1 GEMM instead of 3)
   if hasattr(layer.self_attn, '_qkv_fused') and layer.self_attn._qkv_fused:
       qkv = layer.self_attn.qkv_proj(hidden_states)
       q_out, k_out, v_out = layer.self_attn._qkv_split_sizes
       query_state, key_state, value_state = qkv.split([q_out, k_out, v_out], dim=-1)
   else:
       # Fallback to unfused path
   ```

**Usage** (call after model load, before `torch.compile`):
```python
from openpi.models_pytorch.fusion_utils import fuse_qkv_projections, fuse_gate_up_projections

fuse_qkv_projections(model.paligemma_with_expert.paligemma.language_model)
fuse_qkv_projections(model.paligemma_with_expert.gemma_expert.model)
fuse_gate_up_projections(model.paligemma_with_expert.paligemma.language_model)
fuse_gate_up_projections(model.paligemma_with_expert.gemma_expert.model)
```

**Risk**: Low — weight packing is exact, no approximation

---

## ☐ OPT-8: Reduce Denoising Steps (10→5) (−33ms)

**Target**: `pi0_pytorch.py` — `sample_actions()` `num_steps` parameter

**Implementation**: Simply change `num_steps` parameter at call site:
```python
actions = model.sample_actions(device, observation, num_steps=5)
```

**Validation required**: Compare action trajectories (MSE, max deviation) between 10-step and 5-step on your RBY1-XHand tasks. Flow matching typically converges well at 4-5 steps.

**Risk**: Medium — must validate action quality per-task

---

## OPT-9: Vision Resolution / Token Reduction (−15ms)

### 9A: Resolution Reduction (☐ Pending)
```python
IMAGE_RESOLUTION = (160, 160)  # was (224, 224)
# Tokens: 256 → ~121 per image, ~50% fewer VLM prefix tokens
```

### ✅ 9B: Token Merging (ToMe) — Training-Free

**File**: **NEW** `src/openpi/models_pytorch/tome_utils.py`

**What was done**: Implemented bipartite soft matching that merges similar tokens at each SigLIP encoder layer. Patches each layer's forward to apply merging after attention+MLP.

- Configurable merge rate `r` controls tokens removed per layer
- With 27 SigLIP layers and 256 input tokens:
  - `r=2`: 256→202 tokens (~21% reduction, conservative)
  - `r=4`: 256→148 tokens (~42% reduction, recommended)
  - `r=8`: 256→40 tokens (~84% reduction, aggressive)

**Usage** (call after model load, before inference):
```python
from openpi.models_pytorch.tome_utils import patch_siglip_with_tome

patch_siglip_with_tome(
    model.paligemma_with_expert.paligemma.model.vision_tower.vision_model,
    r=4,  # recommended starting point
)
```

**Risk**: Low-Medium — training-free, but aggressive r values may impact quality. Start with r=2 and increase.

### 9C: Per-Camera Resolution (☐ Pending)
```python
CAMERA_RESOLUTIONS = {
    "head_cam_0": (224, 224),
    "head_cam_1": (224, 224),
    "left_wrist_cam": (160, 160),
    "right_wrist_cam": (160, 160),
}
```

---

## ✅ OPT-10: Channels-Last Memory Format (−3-8ms)

**File**: `src/openpi/models_pytorch/pi0_pytorch.py` — `__init__()` lines 135-140, `embed_prefix()` lines 228-231

**What was done**:

1. Vision tower converted to `channels_last` (NHWC) at init:
   ```python
   # OPT-10: Convert vision tower to channels_last memory format.
   self.paligemma_with_expert.paligemma.model.vision_tower.to(
       memory_format=torch.channels_last
   )
   ```

2. Input images converted to NHWC before `embed_image()`:
   ```python
   # OPT-10: Convert input image to channels_last (NHWC) memory format
   img_cl = img.to(memory_format=torch.channels_last)
   return self.paligemma_with_expert.embed_image(img_cl)
   ```

Enables cuDNN to select faster NHWC convolution kernels for SigLIP's Conv2d patch embedding.

**Risk**: Very Low

---

## ☐ OPT-11: INT8 Weight-Only Quantization (−20-30%)

```python
from torchao.quantization import quantize_, int8_weight_only
quantize_(model, int8_weight_only())
```

**Risk**: Low — weight-only doesn't affect activation precision

---

## ☐ OPT-12: INT8 KV Cache (−5-8ms on memory-bound decode)

Quantize cached K/V to INT8 after prefix prefill. Reduces KV cache memory by 2× → faster memory reads during attention.

**Files**: `pi0_pytorch.py` (sample_actions, denoise_step)
**Risk**: Low — INT8 KV cache is well-established

---

## ☐ OPT-13: 2:4 Structured Sparsity (−15-25%)

Blackwell (RTX 5090 + Jetson Thor) has native 2:4 sparse Tensor Core support.

**Requires**: Short fine-tuning (few hundred steps) to recover accuracy after pruning.
**Files**: New script `scripts/apply_sparsity.py`
**Risk**: Medium — needs fine-tuning, but 2:4 is hardware-native on your platforms

---

## ☐ OPT-14: Structured Pruning with Wanda (−20-40% + retrain)

Remove entire attention heads or MLP neurons to physically shrink model.

**Requires**: Fine-tuning on your RBY1-XHand dataset
**Risk**: Medium-High — must validate thoroughly

---

## ☐ OPT-15: Async Prefill Pipeline (Latency Hiding)

Overlap next observation's VLM prefill with current action execution.

**Files**: New wrapper class or modify ROS inference script
**Risk**: Low — pure scheduling optimization

---

## ☐ OPT-16: Consistency Flow Distillation (10 steps → 1-2 steps)

Train a student that matches the teacher's 10-step output in 1-2 steps.

**Requires**: Training on your dataset, same model architecture
**Expected**: 10-step → 1-2 steps = ~5-10× denoising speedup
**Risk**: Medium — quality depends on distillation quality

---

## ☐ OPT-17: Full Model Distillation (Teacher→Student)

Distill the full 2.6B model to a ~0.6B student.

**Expected speedup**: ~3-4× (2.6B → 0.6B)
**Risk**: High — significant effort, must validate on real robot

---

## ☐ OPT-18: Vision Encoder Replacement

| Option | Change | Speedup | Effort |
|--------|--------|---------|--------|
| SigLIP-2 ViT-B/16 | Fewer tokens (196 vs 256) | ~1.3× | Medium |
| EfficientViT-B1 | 9M vs 87M params | ~5× | High |
| SigLIP + NaFlex | Variable tokens per resolution | ~1.5× avg | Medium |
| MobileSigLIP (distilled) | ~25M params | ~3× | High |

---

## ☐ OPT-19: TensorRT Export (Specific Components)

Export individual components as TensorRT engines.

**Risk**: Medium — ONNX export can be tricky with custom ops

---

## Implementation Priority & Roadmap

### Phase 1: Zero-Risk Quick Wins ✅ COMPLETE
- [x] OPT-3: while→for loop
- [x] OPT-1: SDPA attention
- [x] OPT-10: channels_last
- [x] OPT-6: Graph break elimination

### Phase 2: Compile + Cache (mostly complete)
- [x] OPT-2: Separate compile units
- [x] OPT-4: Static KV cache
- [ ] OPT-5: Batch camera images

### Phase 3: Fusion + Steps (mostly complete)
- [x] OPT-7: QKV/MLP fusion
- [ ] OPT-8: Reduce steps (10→5) + validate

### Phase 4: Quantization (3-5 days)
- [ ] OPT-11: INT8 weight-only (safest)
- [ ] OPT-12: INT8 KV cache
- [ ] OPT-19: TensorRT export (SigLIP first)

### Phase 5: Sparsity + Pruning (1-2 weeks)
- [ ] OPT-13: 2:4 structured sparsity + fine-tune
- [ ] OPT-14: Wanda pruning + fine-tune

### Phase 6: Architecture + Distillation (2-4 weeks)
- [x] OPT-9B: Vision token merging (ToMe)
- [ ] OPT-9A/9C: Vision resolution reduction
- [ ] OPT-15: Async prefill pipeline
- [ ] OPT-16: Consistency flow distillation (10→1-2 steps)
- [ ] OPT-18: Vision encoder replacement
- [ ] OPT-17: Full model distillation (final)

### Projected Cumulative Speedup

| After Phase | Jetson Thor | RTX 5090 |
|-------------|-------------|----------|
| Baseline | 190ms | ~80ms |
| Phase 1 ✅ | ~145ms | ~58ms |
| Phase 2 (current) | ~120ms | ~45ms |
| Phase 3 | ~80ms | ~28ms |
| Phase 4 | ~55ms | ~18ms |
| Phase 5 | ~40ms | ~13ms |
| Phase 6 | ~15-25ms | ~5-10ms |
