# Databricks notebook source
# MAGIC %md
# MAGIC # Paged Attention with vLLM
# MAGIC [vLLM](https://vllm.ai/) is an open-source library that achieves up to 24x higher throughput for serving large language models by using a new attention algorithm called PagedAttention that efficiently manages memory.
# MAGIC
# MAGIC The greatesdt benefits of vLLM are seen when running inference on batches of inputs. Inference on prompts run serially runs at around 27 tps, compared to almost 275 tps for batches. (Note—this does not appear to be order dependent; these throughput numbers remained approximately the same whether the batch or the serial prompts were executed first)

# COMMAND ----------

# MAGIC %pip install --upgrade vllm torch transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from utils import generate_text_vllm, clear_model, torch_profile_to_dataframe
from vllm import LLM, SamplingParams
import huggingface_hub
import pandas as pd
import torch
import transformers
import time

# COMMAND ----------

huggingface_hub.login()

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic Usage

# COMMAND ----------

llm = LLM(model="meta-llama/Llama-2-7b-hf")

# COMMAND ----------

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# COMMAND ----------

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

results = llm.generate(prompts, sampling_params)

# COMMAND ----------

results

# COMMAND ----------

# MAGIC %md
# MAGIC # Throughput and Memory
# MAGIC
# MAGIC ## Serial Prompts

# COMMAND ----------

out = generate_text_vllm(prompts, llm, False, temperature=0.8, top_p=0.95, max_tokens=50)
pd.DataFrame(out)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch prompts

# COMMAND ----------

generate_text_vllm(prompts, llm, True, temperature=0.8, top_p=0.95, max_tokens=50)

# COMMAND ----------

# MAGIC %md
# MAGIC # Torch Profiling -- Basic

# COMMAND ----------

import torch.profiler as profiler

with profiler.profile(
    record_shapes=True,
    profile_memory=True,
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
) as prof:
  output = generate_text_vllm(prompts, llm, True, temperature=0.8, top_p=0.95, max_tokens=50)

torch_profile_to_dataframe(prof).sort_values("Self CUDA %", ascending=False)
