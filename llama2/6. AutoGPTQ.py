# Databricks notebook source
# MAGIC %md
# MAGIC # Quantize models with GPTQ
# MAGIC
# MAGIC Using the [`AutoGPTQ` library](https://github.com/PanQiWei/AutoGPTQ#quick-installation). Compare to the baseline performance [here](https://e2-dogfood.staging.cloud.databricks.com/?o=6051921418418893#notebook/418210139975057).
# MAGIC
# MAGIC This approach resulted in a model memory footprint of 3.9GB and approximately 32 tokens per second throughput.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes and Resources
# MAGIC
# MAGIC **Notes**
# MAGIC As of 2023-08-28, there were still some gotchas and sharp edges with trying to use AutoGPTQ. In particular, following the examples in the [Making LLMs lighter with AutoGPTQ and transformers](https://huggingface.co/blog/gptq-integration#native-support-of-gptq-models-in-%F0%9F%A4%97-transformers) release blog resulted in CUDA out-of-memory errors. To fix this, it was necessary to remove the `device_map="auto"` argument. According to a [recent issue](https://github.com/PanQiWei/AutoGPTQ/issues/291#issuecomment-1695646845) in the AutoGPTQ library, AutoGPTQ automatically uses GPUs correctly for quantization; it appears there are some undesirable interactions with the device mapping from the accelerate library.
# MAGIC
# MAGIC **Docs and Further Reading**
# MAGIC - https://gist.github.com/TheBloke/b47c50a70dd4fe653f64a12928286682#file-quant_autogptq-py
# MAGIC - https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization
# MAGIC - https://huggingface.co/docs/transformers/main_classes/quantization
# MAGIC - https://huggingface.co/docs/transformers/v4.32.1/en/main_classes/quantization#transformers.GPTQConfig
# MAGIC - https://github.com/PanQiWei/AutoGPTQ/issues/179
# MAGIC - https://github.com/PanQiWei/AutoGPTQ/issues/291
# MAGIC - https://github.com/PanQiWei/AutoGPTQ/issues/291#issuecomment-1695992421

# COMMAND ----------

# MAGIC %pip install --upgrade torch transformers accelerate huggingface_hub optimum
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# install the AutoGPTQ Library corresponding to the CUDA version (11.8)
%pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
dbutils.library.restartPython()

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

# COMMAND ----------

huggingface_hub.login()

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", use_cache=True, padding_side="left"
)

quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",
    tokenizer=tokenizer,
    group_size=128,  # default
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Model

# COMMAND ----------

model

# COMMAND ----------

# MAGIC %md
# MAGIC # Save the model
# MAGIC This will also save the quantization config.

# COMMAND ----------

save_folder = "/dbfs/daniel.liden/models/llama2GPTQc4/"
model.save_pretrained(save_folder)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the quantized model
# MAGIC First detach and reattach compute and then reinstall the required libraries. This is to make sure we accurately measure the CUDA memory usage.

# COMMAND ----------

# MAGIC %pip install --upgrade torch transformers accelerate huggingface_hub optimum
# MAGIC %pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from utils import generate_text, clear_model, torch_profile_to_dataframe, wrap_module_with_profiler
import huggingface_hub
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, pipeline, GPTQConfig
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

save_folder = "/dbfs/daniel.liden/models/llama2GPTQc4/"
gptq_config = GPTQConfig(bits=4, disable_exllama=False, use_cuda_fp16=False)
model = AutoModelForCausalLM.from_pretrained(
    save_folder,
    device_map="auto",
    quantization_config=gptq_config,
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", use_cache=True, padding_side="left"
)

# COMMAND ----------

model.config.quantization_config.__dict__

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

df.head(30)

# COMMAND ----------


