# Databricks notebook source
# MAGIC %md
# MAGIC This notebook shows how to use llama.cpp via the [lama-cpp-python](https://github.com/abetlen/llama-cpp-python) package.
# MAGIC
# MAGIC llama.cpp enables model inference in c/c++. Its original goal was to "run the LLaMA model using 4-bit integer quantization on a MacBook" but its scope has since expanded considerably.
# MAGIC
# MAGIC Here, we will use it with CUDA.
# MAGIC
# MAGIC # 1. Install llama.cpp

# COMMAND ----------

# Clone the repo
!git clone https://github.com/ggerganov/llama.cpp /databricks/driver/llama.cpp/
%cd /databricks/driver/llama.cpp

# COMMAND ----------

# install nvidia cuda toolkit
!apt-get install nvidia-cuda-toolkit -y

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build llama.cpp with cuBLAS support

# COMMAND ----------

# MAGIC %%bash
# MAGIC mkdir build
# MAGIC cd build
# MAGIC cmake .. -DLLAMA_CUBLAS=ON
# MAGIC cmake --build . --config Release

# COMMAND ----------

# MAGIC %md
# MAGIC Skip to step 5 if you've already converted and quantized the model and saved the quantized model!
# MAGIC # 2. Download or move the model to the ./models directory of llama.cpp
# MAGIC First, download the model from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf). You can use, e.g.,
# MAGIC
# MAGIC ```
# MAGIC git lfs install
# MAGIC git clone git@hf.co:meta-llama/Llama-2-7b-hf <target_directory>
# MAGIC ```

# COMMAND ----------

# MAGIC %cp -r /dbfs/daniel.liden/models/llama2/ ./models/llama2/

# COMMAND ----------

# install python dependencies
!python3 -m pip install -r requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Convert the model to the gguf format

# COMMAND ----------

!python3 convert.py ./models/llama2/

# COMMAND ----------

# verify we now have a gguf model
!find ./models/llama2/ -name "*.gguf"

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Quantize the model
# MAGIC
# MAGIC You can see the different quantization options with `quantize --help`. Recommendations can be found in [this issue](https://github.com/ggerganov/llama.cpp/discussions/2094) (may not be up to date)

# COMMAND ----------

!./build/bin/quantize --help

# COMMAND ----------

!./build/bin/quantize ./models/llama2/ggml-model-f16.gguf ./models/llama2/ggml-model-q5_k_m.gguf q5_k_m

# COMMAND ----------

# Optionally, save the quantized model to dbfs
import os
source_path = "./models/llama2/ggml-model-q5_k_m.gguf"
target_path = "/daniel.liden/models/ggml-model-q5_k_m.gguf"

dbutils.fs.cp("file:" + os.path.abspath(source_path), target_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Run inference
# MAGIC
# MAGIC Resume here if you've already downloaded and quantized the model!

# COMMAND ----------

dbutils.fs.ls("daniel.liden/models/")

# COMMAND ----------

# MAGIC %cp /dbfs/daniel.liden/models/ggml-model-q5_k_m.gguf ./models/ggml-model-q5_k_m.gguf

# COMMAND ----------

!ls ./models/

# COMMAND ----------

# run the inference
!./build/bin/main -m ./models/ggml-model-q5_k_m.gguf -n 128 -ngl 64 --prompt "The steps to make a good chemex pour-over coffee are as follows:\n1."

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Python bindings with [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
# MAGIC
# MAGIC Once you have a gguf-formatted and quantized model, you can use the high-level Python API offered by `llama-cpp-python` to work with it. See notebook 8a for details.
