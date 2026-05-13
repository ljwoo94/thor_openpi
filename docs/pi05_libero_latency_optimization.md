# pi05_libero Latency Optimization

This document tracks the `pi05_libero` inference latency optimization effort. Every optimization progress update must update this file and be committed, so readers can follow what changed, what was measured, and what still needs verification.

## Goals

- Reduce warm-cache `Policy.infer` P50/P95 latency for `pi05_libero`.
- Keep the first optimization track compatible with existing `pi05_base` / `pi05_libero` checkpoints.
- Treat model architecture changes as a separate research track because they require retraining, distillation, or new checkpoints.
- Use Jetson Thor as the production target and H100 as the fast training/evaluation target.
- Do not accept NVIDIA latency claims from this Apple Silicon workstation. Jetson Thor and H100 measurements must be supplied by a user running on those devices.

## Current Inference Path

For PyTorch checkpoints, the inference path is:

1. `src/openpi/policies/policy.py::Policy.infer`
   - Applies input transforms.
   - Converts numpy inputs to PyTorch tensors.
   - Builds an `Observation`.
   - Calls `model.sample_actions`.
   - Converts outputs back to numpy and applies output transforms.
2. `src/openpi/models_pytorch/pi0_pytorch.py::PI0Pytorch.sample_actions`
   - Preprocesses observation tensors.
   - Embeds image and language prefix tokens.
   - Runs a cached prefix pass through PaliGemma.
   - Runs the action expert denoising loop for `num_steps`.
3. `src/openpi/training/config.py::pi05_libero`
   - Uses `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`.
   - Uses LIBERO transforms and `pi05_base` weights.

For `pi05_libero`, likely latency hotspots are input transforms/tokenization, image prefix encoding, suffix mask/position construction, Gemma expert denoising steps, attention kernels, dtype conversions, and host-device transfers.

## Optimization Tracks

### Track 1: Checkpoint-Safe Inference

These changes must preserve existing checkpoint compatibility.

- Add a benchmark/profiling harness for fixed `pi05_libero` inputs.
- Use `torch.inference_mode()` for PyTorch policy inference.
- Reduce avoidable numpy copies and host-device transfers.
- Cache repeated prompt tokenization where prompts repeat across LIBERO tasks.
- Skip or cache known masked/padded image inputs where output equivalence is preserved.
- Precompute static timestep, mask, position, and sinusoidal embedding values for fixed serving shapes.
- Split compiled regions into stable static pieces: prefix encode and denoise step.
- Compare eager attention, PyTorch SDPA, TorchInductor, CUDA Graph replay, and optional Triton fusions.
- Validate BF16, FP8, and TensorRT/Torch-TensorRT paths separately on Jetson Thor and H100.

### Track 2: Architecture And Retraining Research

These changes are optional and require new weights or fine-tuning.

- Reduce denoising steps and train or distill for fewer-step inference.
- Distill the action expert into a smaller LIBERO-specific expert.
- Prune or compress prefix tokens after image-language encoding.
- Reduce expert depth/width or use lower-rank projections.
- Evaluate FP8-aware training and deployment for Jetson Thor and H100.

## Benchmark Protocol

Primary metric:

- Warm-cache end-to-end `Policy.infer` P50/P95 latency, batch size 1.

Required measurements:

- Total `Policy.infer` latency.
- Input transform latency.
- Host-to-device transfer latency.
- Prefix encode latency.
- Denoising loop latency and per-step breakdown when available.
- Output transform latency.
- Peak GPU memory if available.

Correctness and quality:

- Use fixed observations and fixed noise for action tensor comparisons.
- A roughly 10% action-level difference may be acceptable for experimental FP8 or kernel changes, but rollout behavior is the final quality gate.
- Run representative LIBERO rollouts before marking a change as verified.

NVIDIA verification:

- H100 can be used for fast profiling and evaluation.
- Jetson Thor is required before accepting a production latency claim.
- Apple Silicon local runs are only smoke tests for imports, shape compatibility, and documentation.

## Manual Verification Request Template

When an optimization commit is ready for NVIDIA validation, ask the user to run the benchmark on Jetson Thor and/or H100 and report:

```text
Commit:
Hardware:
GPU driver / CUDA / PyTorch versions:
Command:
Warmup iterations:
Measured iterations:
Policy.infer P50/P95:
Model-only P50/P95:
Transform P50/P95:
Peak GPU memory:
Correctness result:
LIBERO rollout result:
Notes / regressions:
```

## Progress Log

| Date | Commit | Change | Expected Impact | Verification Status | User Feedback |
| --- | --- | --- | --- | --- | --- |
| 2026-05-13 | initial doc commit | Add living optimization plan and progress workflow. | Makes future optimization work auditable and keeps manual NVIDIA verification explicit. | Documentation-only; no NVIDIA verification needed. | Pending. |

## Commit And Update Rule

- Every optimization progress update must update this document.
- Every optimization progress update must be committed.
- If user feedback changes the next step or invalidates an optimization, update the progress log in a follow-up commit.
- Do not mark a GPU optimization as verified until Jetson Thor or H100 measurements are reported by the user.
