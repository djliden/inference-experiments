# Inference Experiments with LLaMA v2 7b

This repository contains a series of notebooks demonstrating various optimizations to make the LLaMA model faster and/or smaller.

[1. Baseline performance](1.%20baseline.ipynb): This notebook establishes the "baseline" performance of the LLama v2 7b model (GPU, `bfloat16`).

[2. Better Transformer](2.%20better_transformer.ipynb): This notebook explores the use of model.to_bettertransformer() for performance improvement.

[3. Torch Compile](3.%20compile.ipynb): This notebook shows the effect of using `torch.compile()` on model performance.

[4. Load in 8 bit](4.%20int8%20Matrix%20Multiplication.ipynb): This notebook demonstrates the effects on model size and performance of performing 8-bit quantization with `bitsandbytes`.

[5. fp4 Mixed-Precision inference](5.%20fp4%20mixed-precision%20inference.ipynb) This notebook shows how to use mixed-precision (FP4) for inference to reduce memory footprint and increase speed.

[6. Quantization with AutoGPTQ](6.%20AutoGPTQ.ipynb): This notebook uses the AutoGPTQ library to quantize the model, reducing the model memory footprint and improving throughput.

[7. VLLM](7.%20VLLM.ipynb): This notebook demonstrates the use of the VLLM library for model optimization, including steps for model conversion and quantization.

[8. Llama.cpp](8.%20llama.cpp.ipynb): This notebook shows how to use llama.cpp for model inference in C/C++, including steps to convert and quantize the model.

[8a. Llama-cpp-python](8a.%20llama-cpp-python.ipynb): This notebook demonstrates how to use the high-level Python API offered by llama-cpp-python to work with the gguf-formatted and quantized model.