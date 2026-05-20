# Thor Precision Sweep

This workflow keeps FP32 ONNX export as the accuracy baseline and only lowers precision in scoped projection and attention paths. Do not use global FP16 export as a baseline for Thor robot validation; it has shown large accuracy loss.

## Export Modes

Baseline FP32:

```bash
python openpi_on_thor/pytorch_to_onnx.py \
  --precision fp32 \
  --compute_dtype fp32
```

Expert BF16 projections:

```bash
python openpi_on_thor/pytorch_to_onnx.py \
  --precision fp32 \
  --compute_dtype fp32 \
  --bf16_expert_projections
```

Expert FP8 QDQ from FP32 export baseline:

```bash
python openpi_on_thor/pytorch_to_onnx.py \
  --precision fp8 \
  --compute_dtype fp32 \
  --bf16_expert_projections \
  --fp8_expert_linears \
  --quantize_attention_matmul
```

Add SigLIP vision encoder after expert variants pass:

```bash
python openpi_on_thor/pytorch_to_onnx.py \
  --precision fp8 \
  --compute_dtype fp32 \
  --bf16_expert_projections \
  --bf16_siglip_encoder_projections \
  --fp8_expert_linears \
  --fp8_siglip_encoder_linears \
  --quantize_attention_matmul \
  --quantize_siglip_attention_matmul
```

Language projection and linears should be enabled after expert and SigLIP variants pass:

```bash
python openpi_on_thor/pytorch_to_onnx.py \
  --precision fp8 \
  --compute_dtype fp32 \
  --bf16_expert_projections \
  --bf16_siglip_encoder_projections \
  --bf16_language_projections \
  --fp8_expert_linears \
  --fp8_siglip_encoder_linears \
  --fp8_language_linears \
  --quantize_attention_matmul \
  --quantize_siglip_attention_matmul
```

## TensorRT Build

Keep `build_engine.sh` usage unchanged, but set precision flags explicitly when testing.

Recommended first TensorRT checks:

```bash
PRECISION_FLAGS="--stronglyTyped" \
ONNX_PATH=/path/to/model_fp32.onnx \
./openpi_on_thor/build_engine.sh
```

```bash
PRECISION_FLAGS="--bf16 --stronglyTyped" \
ONNX_PATH=/path/to/model_fp32_bf16expert.onnx \
./openpi_on_thor/build_engine.sh
```

```bash
PRECISION_FLAGS="--bf16 --fp8 --stronglyTyped" \
ONNX_PATH=/path/to/model_fp8_bf16expert_fp8expert_fp8attn.onnx \
./openpi_on_thor/build_engine.sh
```

`build_engine.sh` respects an explicit `PRECISION_FLAGS` environment value. If it is unset, it falls back to its filename-based default behavior.

## Inspection

Before building TensorRT, inspect the ONNX graph:

```bash
python openpi_on_thor/inspect_onnx_precision.py /path/to/model.onnx
```

Check:

- FP32 baseline has no unexpected `QuantizeLinear` or `DequantizeLinear`.
- FP8 variants have QDQ-adjacent `MatMul` or `Gemm` nodes.
- Sensitive names such as `patch_embedding`, `position_embedding`, `layer_norm`, `input_layernorm`, `post_attention_layernorm`, and `model.norm` do not appear in QDQ warnings.
- Cast count does not grow unexpectedly across variants.

## Sweep Order

1. FP32 ONNX + FP32 TensorRT flags.
2. FP32 ONNX + TensorRT BF16 flags.
3. BF16 expert projections, no FP8.
4. BF16 expert projections + FP8 expert linears.
5. BF16 expert projections + FP8 expert linears + Gemma attention matmul QDQ.
6. Add BF16 SigLIP encoder projections.
7. Add FP8 SigLIP encoder linears.
8. Add SigLIP attention matmul QDQ.
9. Add BF16 language projections.
10. Add FP8 language linears.
11. Try BF16 multimodal projector only after all earlier variants pass.

Keep these FP32 unless a targeted test proves otherwise:

- SigLIP patch embedding and position embedding.
- SigLIP/Gemma/PaliGemma norms.
- RoPE and softmax internals.
- Denoising Euler accumulation.
- Final action output projection.
