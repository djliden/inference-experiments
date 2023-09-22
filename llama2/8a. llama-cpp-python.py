# Databricks notebook source
# MAGIC %md
# MAGIC # Python bindings with [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
# MAGIC
# MAGIC Once you have a gguf-formatted and quantized model, you can use the high-level Python API offered by `llama-cpp-python` to work with it.

# COMMAND ----------

# MAGIC %cd /databricks/driver/
# MAGIC %cp /dbfs/daniel.liden/models/ggml-model-q5_k_m.gguf ./ggml-model-q5_k_m.gguf

# COMMAND ----------

# install nvidia cuda toolkit
!apt-get install nvidia-cuda-toolkit -y

# COMMAND ----------

# Install
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade llama-cpp-python

# COMMAND ----------

# initialize
# make sure to set `n_gpu_layers` to use GPU.
from llama_cpp import Llama
llm = Llama(model_path="./ggml-model-q5_k_m.gguf", n_gpu_layers = -1)

# COMMAND ----------

output = llm("The steps to make a good chemex pour-over coffee are as follows:\n1.", max_tokens=250, echo=True)
output

# COMMAND ----------

# MAGIC %md
# MAGIC ## Throughput and memory

# COMMAND ----------

from utils import generate_text_llama_cpp_py, torch_profile_to_dataframe
import pandas as pd

prompts = [
    "Dreams are",
    "The future of technology is",
    "In a world where magic exists,",
    "The most influential person in history is",
    "One of the most intriguing mysteries of the universe is",
    "When humans finally ventured out into the cosmos, they discovered",
    "The relationship between artificial intelligence and humanity has always been",
    "As the boundaries of science and fiction blur, the implications for society become",
    "In the depths of the enchanted forest, ancient creatures and forgotten tales come to life, revealing",
    "While many believe that technological advancements will be the key to solving humanity's greatest challenges, others argue that it will only exacerbate existing inequalities, leading to"
]

# COMMAND ----------

out = generate_text_llama_cpp_py(prompts, model=llm, batch=False,
              max_tokens=50)
pd.DataFrame(out)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Torch Profiling
# MAGIC

# COMMAND ----------

import torch.profiler as profiler

with profiler.profile(
    record_shapes=True,
    profile_memory=True,
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
) as prof:
  output = generate_text_llama_cpp_py(prompts, model=llm, batch=False,
              max_tokens=50)

torch_profile_to_dataframe(prof).sort_values("Self CUDA %", ascending=False)

# COMMAND ----------


