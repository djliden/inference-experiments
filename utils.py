from typing import Optional
import tokenizers
from typing import Callable
import time
from transformers import pipeline
import torch
import pandas as pd
from io import StringIO
#import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function


def estimate_tps(
    tokenizer: tokenizers.Tokenizer,
    output: str,
    time_s: float,
    input_text: Optional[str] = None,
) -> float:
    input_len = len(tokenizer(input_text)["input_ids"]) if input_text else 0
    output_len = len(tokenizer(output)["input_ids"])

    return (output_len - input_len) / time_s
  
# make it a decorator~
def tps_decorator(tokenizer: tokenizers.Tokenizer, input: Optional[str] = None):
    def inner_decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            output_text = result[0]["generated_text"]
            tps_value = estimate_tps(tokenizer, output_text, end_time-start_time, input)
            
            print(f'tps: {tps_value}')
            return result
        return wrapper
    return inner_decorator


import time

def generate_text(input_text, model, tokenizer, batch=False, **kwargs):
    # Convert input to list if it's a string
    if isinstance(input_text, str):
        input_text = [input_text]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_tensors=True,
        **kwargs
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    results = []

    if batch:
        # Process all input texts in a single batch
        start_time = time.time()
        output = pipe(input_text)
        end_time = time.time()
        input_tokens = sum(len(tokenizer.tokenize(t)) for t in input_text)
        output_tokens = sum(len(t[0]["generated_token_ids"]) for t in output)

        tokens_per_second = (output_tokens - input_tokens) / (end_time - start_time)

        max_memory = torch.cuda.max_memory_allocated() / 1024.0**3 if torch.cuda.is_available() else None

        results = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second,
            "elapsed_time": end_time - start_time,
            "max_cuda_memory": max_memory,
        }

    else:
        results = {
            "input_tokens": [],
            "total_tokens": [],
            "tokens_per_second": [],
            "elapsed_time": [],
            "max_cuda_memory": [],
            "output_text": [],
        }
        for text in input_text:
            start_time = time.time()
            output = pipe([text])
            end_time = time.time()

            input_tokens = len(tokenizer.tokenize(text))
            output_tokens = len(output[0][0]["generated_token_ids"])
            output_text = tokenizer.decode(output[0][0]["generated_token_ids"],
                                           skip_special_tokens=True)

            tokens_per_second = (output_tokens - input_tokens) / (end_time - start_time)

            max_memory = torch.cuda.max_memory_allocated() / 1024.0**3 if torch.cuda.is_available() else None

            results["input_tokens"].append(input_tokens)
            results["total_tokens"].append(output_tokens)
            results["tokens_per_second"].append(tokens_per_second)
            results["elapsed_time"].append(end_time - start_time)
            results["max_cuda_memory"].append(max_memory)
            results["output_text"].append(output_text)
    # cleanup
    del(pipe)
    return results

    

def profile_generate_text(input_text, model, tokenizer, **kwargs):
    if isinstance(input_text, str):
        input_text = [input_text]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_tensors=True,
        **kwargs
    )

    return None

def clear_model(model, tokenizer):
    del model
    del tokenizer
    del pipeline
    torch.cuda.empty_cache()

    return None

import pandas as pd
from io import StringIO
import re

def torch_profile_to_dataframe(prof):
    prof_output = prof.key_averages().table(sort_by="self_cuda_time_total")
    lines = [line for line in prof_output.split('\n') if not line.startswith('-')]

    # Convert the processed string back to a single string
    processed_str = '\n'.join(lines[:-3])  # omitting the last two lines with totals

    # Convert the string to a DataFrame
    df = pd.read_csv(StringIO(processed_str), sep="\s\s+", engine='python')

    # Handle percent columns
    percent_cols = ['Self CPU %', 'CPU total %', 'Self CUDA %']
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].str.rstrip('%').astype('float') / 100.0

    # Handle time columns
    time_cols = ['Self CPU', 'CPU total', 'CPU time avg', 'Self CUDA', 'CUDA total', 'CUDA time avg']
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].apply(_parse_time)

    return df

def _parse_time(time_str):
    # Parse time values with unit suffixes (e.g., ms, us, s) to microseconds
    time_str = time_str.strip()
    if time_str.endswith('us'):
        return float(time_str.rstrip('us'))
    elif time_str.endswith('ms'):
        return float(time_str.rstrip('ms')) * 1e3
    elif time_str.endswith('s'):
        return float(time_str.rstrip('s')) * 1e6
    else:
        return float(time_str)
      


def wrap_module_with_profiler(module, parent_name=""):
    # This is the key: Only wrap modules that are not containers like nn.ModuleList or nn.Sequential
    if not list(module.children()): 
        original_forward = module.forward

        def new_forward(*args, **kwargs):
            with record_function(parent_name + module.__class__.__name__):
                return original_forward(*args, **kwargs)

        module.forward = new_forward
    else:
        # Recursively wrap children
        for name, child in module.named_children():
            wrap_module_with_profiler(child, parent_name + name + '.')

