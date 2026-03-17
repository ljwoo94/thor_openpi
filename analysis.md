# OpenPI TensorRT Mixed-Precision Optimization Pipeline

This document details the optimizations and insights discovered while compiling the π₀.5 vision-language model into a TensorRT engine using NVIDIA `modelopt` for FP8 and NVFP4 quantization.

## 1. The Core Issue: The `65,504` Overflow Bug

### Problem
The initial compilation pipeline used the `--fp16` and `--fp8` flags in `trtexec`, while the base PyTorch ONNX export was converting tensors via `model = model.to(torch.float16)`.
This approach worked fundamentally, but caused the inference engine's accuracy to completely collapse (yielding a cosine similarity of `< 0.3` to the baseline PyTorch model).

### Root Cause
PaliGemma (the architecture underlying π₀.5) frequently produces activation values and attention logits that exceed `65,504`.
*   Standard `FP16` (Float16) has a hard mathematical ceiling of `~65,504`.
*   When TensorRT attempted to execute unquantized memory layers (like `RMSNorm`, `Softmax`, or Residual additions) in `FP16`, the massive activation values overflowed into `NaN` or `Inf`.
*   This broken activation stream was then fed into the highly-optimized FP8 Tensor Cores, generating pure garbage output and destroying accuracy.

## 2. The Solution: `BFloat16` (BF16) + `FP8`

To preserve the accuracy of the model while maintaining maximum GPU speed, the pipeline was updated to use a combination of Brain Float 16 (`BFloat16`) and `FP8`.

*   **BFloat16** uses the same 16-bit payload size as `FP16` (meaning it runs exactly as fast on the memory bus), but it allocates those bits differently. It has the same massive dynamic exponent range as `FP32` ($10^{38}$), meaning it has no hard limit at `65K` and **cannot overflow** on PaliGemma's large activations.

### Implementation changes
1.  **PyTorch Exporter (`pytorch_to_onnx.py`):**
    We explicitly exported the ONNX blueprint in `FP32` rather than `BFloat16`. This intentionally bypassed a known Hugging Face / PyTorch bug where tracing Llama-style Rotary Position Embeddings (RoPE) in `BFloat16` crashes the exporter by illegally upcasting complex numbers to 128-bit `ComplexDouble`.

2.  **Engine Compilation (`build_engine.sh`):**
    We replaced the `--fp16` flag with `--bf16`. 
    TensorRT now automatically ingests our massive `FP32` ONNX graph, targets the Heavy Matrix operations for `FP8`, and safety downcasts all the unquantized memory operations (Softmax, Norms) into `BF16`. 
    **Result: Cosine similarity restored to `0.99+` at maximum Tensor Core speed.**

3.  **Inference Drivers (`trt_model_forward.py`):**
    The Python runtime scripts packaging the images, noise, and tokens before sending them to the TensorRT bindings were updated to cast tensors to `torch.bfloat16` instead of `torch.float16`, preventing the clipping bug from occurring inside PyTorch before it even reached the GPU.

## 3. Demystifying NVIDIA `modelopt` and Graph Quantization

### "Fake Quantization" vs Hardware Compilation
A major point of confusion resolved in this analysis was understanding what `modelopt` actually does.

*   `modelopt` **does not** permanently change all PyTorch layers into integer representations.
*   It places `QuantizeLinear` and `DequantizeLinear` (Q/DQ) structural markers strictly around computationally heavy nodes (`nn.Linear`, `nn.Conv2d`, `torch.matmul`).
*   It intentionally leaves bandwidth-bound layers (Norms, Activations, Softmax) completely untouched.

When passing this ONNX file to `trtexec`, TensorRT does not run the "quantize and dequantize calculations" linearly. Instead, through **Graph Fusion**, TensorRT permanently deletes the Q/DQ nodes from the software graph and compiles their scaling factors directly into the FP8 Tensor Core hardware instructions. 

During live engine inference, the data flows like this:
1.  `RMSNorm` calculates in memory-speed `BFloat16`.
2.  The FP8 Tensor Core grabs the `BF16` activation, applies the structural FP8 scale factor natively in silicon, rips through the matrix multiplication in pure 8-bit math, and instantly writes the accumulated result back to GPU memory as `BFloat16`.

### The NVFP4 Edge Case
When enabling `NVFP4` (the ultra-low 4-bit precision format for LLM blocks), our engine actually ran *slower* because the ONNX graph was locked to `FP32`.
The highly compressed 4-bit NVFP4 tensors were spending massive amounts of time decompressing back into 32-bit floats for the residual streams, choking the GPU memory bus.

**Resolution:** We manually patched `pytorch_to_onnx.py` by setting `module.input_quantizer._trt_high_precision_dtype = "BFloat16"`. This successfully instructed TensorRT to decompress the NVFP4 blocks directly into fast `BF16` instead of `FP16` or `FP32`.

## 4. Future Opportunities: Expanding FP8 

Currently, `modelopt` is using its `mtq.FP8_DEFAULT_CFG`, which conservatively quantizes only `Linear` and `MatMul` operations.

If you wish to push performance further, TensorRT actually supports FP8 execution for several other layers that are currently falling back to `BF16`.

**Operations you can force into FP8/INT8:**
1.  **KV Cache Engine (`past_key_values`):**
    The biggest memory bottleneck in Vision-Language Models is storing the massive KV cache. High-end frameworks like vLLM natively quantize the KV cache to FP8. You can enable this in TensorRT-LLM, allowing you to run much longer image context windows or higher micro-batch sizes.
2.  **`nn.Conv2d` (Vision Patches):**
    Notice in `pytorch_to_onnx.py` that `mtq` explicitly disabled Convolution quantization: 
    ```python
    quant_cfg["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}
    ```
    The SigLIP vision tower inside PaliGemma relies heavily on an initial 2D Convolution to slice the images into embeddings. While this is a small portion of the network, enabling FP8 on `Conv2d` can yield slight speedups on the Vision encoder pass if calibration data proves it doesn't degrade accuracy.
