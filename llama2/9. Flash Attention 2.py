# Databricks notebook source
# MAGIC %md
# MAGIC # Flash Attention 2
# MAGIC - Running on g5.4xlarge instance
# MAGIC - Uses the new native flash attention 2 integration from the Hugging Face Transformers library
# MAGIC - marginally faster than baseline; fraction of a token per second.

# COMMAND ----------

# MAGIC %pip install --upgrade torch git+https://github.com/huggingface/transformers accelerate huggingface_hub flash-attn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from utils import generate_text, clear_model, torch_profile_to_dataframe
import huggingface_hub
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, pipeline
import os
import datetime
import time
import accelerate



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

huggingface_hub.login()

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", use_cache=True, padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    use_cache=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    use_flash_attention_2=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Model

# COMMAND ----------

model

# COMMAND ----------

# MAGIC %md
# MAGIC # Throughput and Memory
# MAGIC
# MAGIC ## Serial Prompts

# COMMAND ----------

out = generate_text(prompts, model, tokenizer, batch=False,
              eos_token_id=tokenizer.eos_token_id, max_new_tokens=50)
pd.DataFrame(out)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch prompts

# COMMAND ----------

out = generate_text(prompts, model, tokenizer, batch=True,
              eos_token_id=tokenizer.eos_token_id, max_new_tokens=50)
out

# COMMAND ----------

# MAGIC %md
# MAGIC # Torch Profiling -- Basic

# COMMAND ----------

import torch.profiler as profiler

with profiler.profile(
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
) as prof:
  output = generate_text(prompts, model, tokenizer, eos_token_id=tokenizer.eos_token_id,
                         max_new_tokens=10)

torch_profile_to_dataframe(prof).sort_values("Self CUDA %", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualize the PyTorch Trace (single token)

# COMMAND ----------

import torch.profiler as profiler

with profiler.profile(
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    #on_trace_ready=torch.profiler.tensorboard_trace_handler('./tmp/log/llamalog'),
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
) as prof:
  output = generate_text("The greatest innovations in technology in the 21st century were", model, tokenizer,
                         eos_token_id=tokenizer.eos_token_id,
                         max_new_tokens=1)

# COMMAND ----------

prof.export_chrome_trace("/dbfs/<path>/baseline_trace.json")

# COMMAND ----------

# MAGIC %md
# MAGIC We can visualize the trace by following these steps:
# MAGIC 1. Download the trace json file to the local machine. From the command line on the local machine: `databricks fs cp dbfs:<path>/trace.json ~/Downloads/baseline_trace.json`
# MAGIC 2. Open the trace file with Google Chrome via `chrome://tracing`

# COMMAND ----------


