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

The checkpoint-safe track is implemented in ordered phases. Each phase must update the progress log and include manual verification instructions when GPU measurements are needed.

#### Phase 0: Baseline Harness

Purpose: make every later optimization measurable.

- Add a local benchmark script for `pi05_libero` that loads a policy with `policy_config.create_trained_policy`.
- Support `--config-name`, `--checkpoint-dir`, `--device`, `--warmup-iters`, `--iters`, `--num-steps`, `--seed`, and optional output file arguments.
- Generate deterministic LIBERO-shaped observations using `libero_policy.make_libero_example`, fixed prompt text, and fixed optional noise.
- Synchronize CUDA before and after timed regions when the device is CUDA.
- Report at minimum local end-to-end `Policy.infer` latency and model-reported `policy_timing["infer_ms"]`.
- Record hardware context: `torch.__version__`, CUDA availability, GPU name, CUDA capability, device string, and whether the loaded checkpoint is PyTorch.
- Acceptance gate: script runs with `--help` locally and runs end-to-end on H100/Jetson with a real checkpoint.

#### Phase 1: Low-Risk Policy Path Cleanup

Purpose: remove Python and data-movement overhead without changing model math.

- Wrap PyTorch inference in `torch.inference_mode()` in `Policy.infer`.
- Replace eager `np.array(x)` conversion with `np.asarray(x)` and `torch.as_tensor` where it avoids copies while preserving behavior.
- Ensure tensor conversion handles scalars, numpy arrays, and existing tensor-like leaves consistently.
- Add optional timing keys around input transforms, tensor conversion, sample_actions, CPU copy, and output transforms.
- Keep output structure unchanged for clients.
- Acceptance gate: fixed-noise actions match baseline within small numerical tolerance; benchmark shows no regression.

#### Phase 2: Transform And Tokenization Cache

Purpose: avoid repeated CPU work in LIBERO serving.

- Add a prompt-token cache for inference when prompt string, discrete-state setting, tokenizer max length, and state inclusion policy are unchanged.
- For `pi05_libero`, note `discrete_state_input=False`, so tokenized prompt does not include state and can be reused across repeated task prompts.
- Keep cache bounded or per-policy-instance to avoid unbounded memory growth.
- Measure transform-only latency before and after.
- Acceptance gate: repeated prompt inputs produce identical token arrays and masks; changing prompt invalidates cache.

#### Phase 3: Static Tensor Precomputation

Purpose: reduce per-denoise-step small tensor allocation and graph breaks.

- Precompute fixed timestep schedule for default `num_steps=10`.
- Precompute suffix attention masks for fixed `action_horizon=10`.
- Precompute suffix causal/AR masks and reusable position increments.
- Precompute sinusoidal embedding frequency basis and compute only the timestep-dependent sine/cosine values per step.
- Avoid repeated construction of Python lists and `torch.tensor(...)` inside `embed_suffix` and `denoise_step` for fixed-shape inference.
- Acceptance gate: action differences remain within tolerance; memory allocation count and per-step latency decrease on CUDA.

#### Phase 4: Compile And Static Graph

Purpose: make TorchInductor optimize stable regions.

- Split compilation into explicit static functions instead of compiling the entire `sample_actions` method first.
- Compile prefix embedding/prefix pass separately from one denoise step.
- Keep `num_steps`, action horizon, action dim, and max token length static for the optimized path.
- Compare `torch.compile` modes:
  - H100: `max-autotune` and `reduce-overhead`.
  - Jetson Thor: `reduce-overhead` first, then `max-autotune` only if compile time and memory are acceptable.
- Add a runtime fallback to the existing eager path.
- Acceptance gate: compiled path warms up successfully and improves warm-cache P50/P95 on target GPU.

#### Phase 5: Attention And Kernel Experiments

Purpose: optimize the real GPU hot spots after profiling confirms them.

- Keep the patched Gemma/PaliGemma attention backend on eager unless a dedicated dtype-safe replacement is implemented.
- Do not use the generic PyTorch SDPA backend in this repo: user validation showed it fails under compiled `sample_actions`
  with an attention bias dtype mismatch in the patched BF16 stack.
- Test Flash-style attention only where supported by installed PyTorch/Transformers/runtime and after mask/bias dtype handling is explicit.
- Profile adaRMSNorm, gated residual, MLP, action projection, and attention kernels.
- Implement custom Triton only after profiling shows a specific unfused operation is hot on H100 or Jetson Thor.
- Candidate Triton/fused kernels:
  - RMSNorm / adaRMSNorm with scale, shift, and gate.
  - gated residual plus dtype cast.
  - small action/time projection blocks.
  - suffix mask construction if still materialized dynamically.
- Acceptance gate: each kernel has baseline-vs-optimized correctness and latency numbers, plus fallback for unsupported hardware.

#### Phase 6: Jetson Thor Deployment Path

Purpose: separate production readiness from H100-only speedups.

- Re-run the benchmark suite on Jetson Thor after every CUDA optimization.
- Validate BF16, FP8, and TensorRT/Torch-TensorRT separately.
- Track compile warmup time, steady-state latency, peak memory, and any unsupported operation.
- Accept TensorRT/Torch-TensorRT only if conversion preserves behavior and reduces steady-state P50/P95 after warmup.
- Acceptance gate: Jetson Thor measurement is recorded in this document before production claims are made.

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

## Detailed Implementation Order

1. Benchmark core model math first: prefix VLM forward and Gemma expert denoise-step forward.
2. Keep eager attention for the PyTorch path; pursue attention speed through TensorRT/QDQ/plugin paths instead of generic SDPA.
3. Measure denoising-step count sensitivity with `--num-steps 10`, `8`, `6`, and `5`; accept only with rollout quality.
4. Split and tune static compiled regions around prefix forward and a single denoise step.
5. Add custom Triton kernels only for measured hot spots inside attention, adaRMSNorm, MLP, or residual math.
6. Evaluate FP8 and TensorRT/Torch-TensorRT on Jetson Thor and H100.
7. Start architecture/distillation track after checkpoint-safe compute optimizations plateau.

Python-level caching and input transform work is secondary. Do not add more conditionals inside denoising or VLM hot paths unless the branch is outside the compiled/static function boundary.

## Current Implementation Backlog

| Priority | Item | Files | Validation |
| --- | --- | --- | --- |
| P0 | Baseline benchmark script | `scripts/benchmark_pi05_libero_latency.py` | `--help` locally; full run on H100/Jetson |
| P0 | Progress doc update on every change | `docs/pi05_libero_latency_optimization.md` | Commit includes doc update |
| P1 | `torch.inference_mode()` and timing breakdown | `src/openpi/policies/policy.py` | Fixed-noise equivalence and timing output |
| P1 | Avoid redundant numpy copies | `src/openpi/policies/policy.py` | Shape/dtype tests and benchmark |
| P2 | Prompt-token cache | `src/openpi/transforms.py` or policy-local wrapper | Repeated prompt cache hit test |
| P2 | Static suffix masks/timestep schedule | `src/openpi/models_pytorch/pi0_pytorch.py` | Fixed-noise equivalence and allocation reduction |
| P2 | Attention acceleration | TensorRT/QDQ/plugin path | Avoid generic SDPA unless patched mask/bias dtypes are fixed |
| P3 | Compile-region split | `src/openpi/models_pytorch/pi0_pytorch.py` | H100 and Jetson P50/P95 |
| P3 | Attention backend experiments | PyTorch Gemma/PaliGemma config path | Correctness and benchmark |
| P4 | Triton fused kernels | new or model-local kernel module | Kernel-level and end-to-end benchmark |
| P4 | TensorRT/Torch-TensorRT path | deployment script/config | Jetson Thor production benchmark |

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

Current baseline command for H100 or Jetson Thor:

```bash
uv run python scripts/benchmark_pi05_libero_latency.py \
  --config-name pi05_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
  --device cuda \
  --warmup-iters 10 \
  --iters 100 \
  --num-steps 10 \
  --output-json /tmp/pi05_libero_latency.json
```

Current compile-mode comparison commands:

```bash
uv run python scripts/benchmark_pi05_libero_latency.py \
  --config-name pi05_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
  --device cuda \
  --warmup-iters 10 \
  --iters 100 \
  --num-steps 10 \
  --disable-compile \
  --output-json /tmp/pi05_libero_latency_no_compile.json

uv run python scripts/benchmark_pi05_libero_latency.py \
  --config-name pi05_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
  --device cuda \
  --warmup-iters 10 \
  --iters 100 \
  --num-steps 10 \
  --compile-mode reduce-overhead \
  --output-json /tmp/pi05_libero_latency_reduce_overhead.json

uv run python scripts/benchmark_pi05_libero_latency.py \
  --config-name pi05_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
  --device cuda \
  --warmup-iters 10 \
  --iters 100 \
  --num-steps 10 \
  --compile-mode max-autotune \
  --output-json /tmp/pi05_libero_latency_max_autotune.json
```

## Progress Log

| Date | Commit | Change | Expected Impact | Verification Status | User Feedback |
| --- | --- | --- | --- | --- | --- |
| 2026-05-13 | initial doc commit | Add living optimization plan and progress workflow. | Makes future optimization work auditable and keeps manual NVIDIA verification explicit. | Documentation-only; no NVIDIA verification needed. | Pending. |
| 2026-05-13 | `d1f846b` | Expand implementation-ready optimization phases and add baseline benchmark utility. | Enables reproducible H100/Jetson measurements before changing model code. | `py_compile` passed locally. Full CLI/runtime check is blocked on Apple Silicon because the project pins `jax[cuda12]`; NVIDIA manual verification required. | Pending. |
| 2026-05-13 | `dc18d99` | Add policy-level timing breakdown and `torch.inference_mode()` for PyTorch inference. | Separates transform, tensor conversion, model, output conversion, output transform, and total latency while removing autograd overhead. | `py_compile` and `git diff --check` passed locally. NVIDIA benchmark required for latency and correctness. | Pending. |
| 2026-05-13 | `1d60cf9` | Cache prompt-only tokenization in `TokenizePrompt`. | Reduced CPU transform work, but user verification showed no meaningful model inference gain. | Reverted in follow-up because VLM/denoise compute dominates. | Low impact. |
| 2026-05-13 | `44890b7` | Precompute pi05 action suffix masks, position offsets, and timestep embedding basis. | Reduced small tensor construction, but user verification showed no meaningful model inference gain. | Reverted in follow-up; avoid extra hot-path conditionals before TensorRT/compile work. | Low impact. |
| 2026-05-13 | `5c5fc42` | Move fixed attention backend setup out of the inference hot path. | Removes repeated config mutation from prefix encode and every denoise step, reducing Python side effects before compile-region work. | `py_compile` and `git diff --check` passed locally. NVIDIA benchmark required for latency and correctness. | Pending. |
| 2026-05-14 | `3502880` | Add PyTorch attention backend override for eager-vs-SDPA core forward benchmarking. | Targeted VLM and denoise attention math. | Invalidated by user verification: SDPA fails under common compiled `sample_actions` due to attention bias/query dtype mismatch in the patched BF16 stack. Reverted in follow-up. | Do not use generic SDPA. |
| 2026-05-14 | `e962ad1` | Replace tensor-valued denoising `while` loop with static integer `for range(num_steps)`. | Preserves the timestep schedule while making the repeated denoise forward path friendlier to TorchInductor and TensorRT graph capture. | `py_compile` and `git diff --check` passed locally. Fixed-noise H100/Jetson correctness and latency benchmark required. | Pending. |
| 2026-05-14 | `8ff9c64` | Add PyTorch compile-mode override for no-compile, `reduce-overhead`, and `max-autotune` benchmarks. | Measures graph conversion cost and steady-state latency explicitly on H100 and Jetson Thor instead of assuming one compile mode is best. | `py_compile` and `git diff --check` passed locally. H100/Jetson compile-mode benchmark required. | Pending. |
| 2026-05-14 | `5acb667` | Add ModelOpt/ONNX QDQ coverage reports and batch all camera images through one vision-tower forward. | QDQ coverage reporting is useful, but the batched vision change blocked compile conversion. | Keep QDQ reporting; revert batched vision prefix in follow-up. | Compile conversion failed for batched vision path. |
| 2026-05-14 | `b834a7b`, `b36abc1` | Revert low-impact prompt token cache and suffix tensor precomputation. | Keeps optimization work focused on core VLM forward, denoise loop, attention kernels, quantization coverage, and TensorRT graph conversion. | Reverts completed. Local checks pending for this batch. | Requested after user validation. |
| 2026-05-14 | `6690d5e` | Make Thor ONNX export denoise loop static and remove export hot-path attention config mutation. | Improves TensorRT/ONNX graph conversion for the real repeated denoise forward instead of Python-side preprocessing. | `py_compile` and `git diff --check` passed locally. Thor export and TensorRT build required. | Pending. |
| 2026-05-14 | `188c6aa` | Revert batched vision prefix while keeping QDQ coverage, and add static-shape ONNX export option. | Avoids compile-breaking prefix logic and gives TensorRT a fixed-shape export path for stronger engine optimization. | `py_compile` and `git diff --check` passed locally. Thor static-shape export and engine build required. | Requested after compile conversion failure. |
| 2026-05-14 | pending | Remove unusable SDPA benchmark/config switch and keep eager attention in PyTorch path. | Prevents invalid benchmark paths and keeps attention optimization focused on QDQ/TensorRT/plugin routes. | `py_compile` and `git diff --check` passed locally. | Requested after SDPA dtype failure. |

## Commit And Update Rule

- Every optimization progress update must update this document.
- Every optimization progress update must be committed.
- If user feedback changes the next step or invalidates an optimization, update the progress log in a follow-up commit.
- Do not mark a GPU optimization as verified until Jetson Thor or H100 measurements are reported by the user.
