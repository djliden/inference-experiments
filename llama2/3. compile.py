# Databricks notebook source
# MAGIC %md
# MAGIC # torch.compile
# MAGIC
# MAGIC Using [`torch.compile()`](https://huggingface.co/docs/transformers/perf_torch_compile). Compare to the baseline performance [here](https://e2-dogfood.staging.cloud.databricks.com/?o=6051921418418893#notebook/418210139975057).
# MAGIC
# MAGIC Appears to result in little to no improvement in tokens per second.

# COMMAND ----------

# MAGIC %pip install --upgrade torch transformers accelerate huggingface_hub
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from utils import generate_text, clear_model, torch_profile_to_dataframe, wrap_module_with_profiler
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
)
model=torch.compile(model, mode="reduce-overhead")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Model

# COMMAND ----------

model

# COMMAND ----------

# MAGIC %md
# MAGIC # Throughput and Memory

# COMMAND ----------

out = generate_text(prompts, model, tokenizer, batch=False,
              eos_token_id=tokenizer.eos_token_id, max_new_tokens=50)

# COMMAND ----------

pd.DataFrame(out)

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
  output = generate_text(prompts, model, tokenizer, eos_token_id=tokenizer.eos_token_id,
                         max_new_tokens=10)

torch_profile_to_dataframe(prof).sort_values("Self CUDA %", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Torch Profiling: by model layer (Work in Progress)

# COMMAND ----------

with torch.autograd.profiler.profile(record_shapes=True, use_cuda=torch.cuda.is_available()) as prof:
    outputs = generate_text(
        prompts[0:2],
        model,
        tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=5,
    )

# COMMAND ----------

df = torch_profile_to_dataframe(prof).sort_values("Self CUDA %", ascending=False)

# Extract layer type using regex
df['Layer Type'] = df['Name'].str.extract(r'model\.layers\.\d+\.(.+)')

# Create an aggregation dictionary for all columns except 'Name' and 'Layer Type'
agg_dict = {col: 'mean' for col in df.columns if col not in ['Name', 'Layer Type']}

# Group by Layer Type and aggregate
grouped = df.groupby('Layer Type').agg(agg_dict).reset_index()

grouped

# COMMAND ----------

df.head(30)

# COMMAND ----------


