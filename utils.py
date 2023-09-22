import re
import time
from io import StringIO
from typing import Callable, Optional


import pandas as pd
import tokenizers
import torch
from torch.profiler import profile, record_function
from transformers import pipeline


def generate_text(input_text, model, tokenizer, batch=False, **kwargs):
    """
    Generate text using a pre-trained language model.

    Args:
        input_text (str or List[str]): The input text or a list of input texts to generate text from.
        model: The pre-trained language model.
        tokenizer: The tokenizer used to preprocess the input text.
        batch (bool, optional): Whether to process all input texts in a single batch. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the text generation pipeline.

    Returns:
        dict: A dictionary containing the following information:
            - input_tokens (int or List[int]): The number of input tokens or a list of input token counts for each input text.
            - output_tokens (int or List[int]): The number of output tokens or a list of output token counts for each input text.
            - tokens_per_second (float or List[float]): The number of tokens generated per second or a list of token generation rates for each input text.
            - elapsed_time (float or List[float]): The elapsed time in seconds or a list of elapsed times for each input text.
            - max_cuda_memory (float or None): The maximum CUDA memory allocated, if available, or None.
            - output_text (str or List[str]): The generated output text or a list of output texts for each input text.
    """
    # Convert input to list if it's a string
    if isinstance(input_text, str):
        input_text = [input_text]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_tensors=True,
        **kwargs,
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
        output_text = [tokenizer.decode(
                t[0]["generated_token_ids"], skip_special_tokens=True) for t in output
            ]

        tokens_per_second = (output_tokens - input_tokens) / (
            end_time - start_time
        )

        max_memory = (
            torch.cuda.max_memory_allocated() / 1024.0**3
            if torch.cuda.is_available()
            else None
        )

        results = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second,
            "elapsed_time": end_time - start_time,
            "max_cuda_memory": max_memory,
            "output_text": output_text,
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
            output_text = tokenizer.decode(
                output[0][0]["generated_token_ids"], skip_special_tokens=True
            )

            tokens_per_second = (output_tokens - input_tokens) / (
                end_time - start_time
            )

            max_memory = (
                torch.cuda.max_memory_allocated() / 1024.0**3
                if torch.cuda.is_available()
                else None
            )

            results["input_tokens"].append(input_tokens)
            results["total_tokens"].append(output_tokens)
            results["tokens_per_second"].append(tokens_per_second)
            results["elapsed_time"].append(end_time - start_time)
            results["max_cuda_memory"].append(max_memory)
            results["output_text"].append(output_text)
    # cleanup
    del pipe
    return results

def generate_text_vllm(input_text, model, batch=False, **kwargs):
    
    # lazy import
    from vllm import LLM, SamplingParams
    
    # Convert input to list if it's a string
    if isinstance(input_text, str):
        input_text = [input_text]
    
    # configure sampling parameters
    sampling_params = SamplingParams(**kwargs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    results = []

    if batch:
        # Process all input texts in a single batch
        start_time = time.time()
        output = model.generate(input_text, sampling_params)
        end_time = time.time()
        input_tokens = sum(len(t.prompt_token_ids) for t in output)
        output_tokens = sum(len(t.outputs[0].token_ids) for t in output)
        output_text = [t.outputs[0].text for t in output]

        tokens_per_second = (output_tokens) / (
            end_time - start_time
        )

        max_memory = (
            torch.cuda.max_memory_allocated() / 1024.0**3
            if torch.cuda.is_available()
            else None
        )

        results = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second,
            "elapsed_time": end_time - start_time,
            "max_cuda_memory": max_memory,
            "output_text": output_text,
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
            output = model.generate(text, sampling_params)
            end_time = time.time()

            input_tokens = len(output[0].prompt_token_ids)
            output_tokens = len(output[0].outputs[0].token_ids)
            output_text = output[0].outputs[0].text

            tokens_per_second = (output_tokens) / (
                end_time - start_time
            )

            max_memory = (
                torch.cuda.max_memory_allocated() / 1024.0**3
                if torch.cuda.is_available()
                else None
            )

            results["input_tokens"].append(input_tokens)
            results["total_tokens"].append(output_tokens)
            results["tokens_per_second"].append(tokens_per_second)
            results["elapsed_time"].append(end_time - start_time)
            results["max_cuda_memory"].append(max_memory)
            results["output_text"].append(output_text)
    # cleanup
    return results

def generate_text_llama_cpp_py(input_text, model, batch=False, **kwargs):
    # Convert input to list if it's a string
    if isinstance(input_text, str):
        input_text = [input_text]
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    results = []

    if batch:
        raise("Batch support not yet enabled; see https://github.com/ggerganov/llama.cpp/pull/3228")
        # Process all input texts in a single batch
        # start_time = time.time()
        # output = model(input_text, **kwargs)
        # end_time = time.time()
        # input_tokens = sum(output["usage"]["prompt_tokens"] for t in output)
        # output_tokens = sum(len(t.outputs[0].token_ids) for t in output)
        # output_text = [t.outputs[0].text for t in output]

        # tokens_per_second = (output_tokens) / (
        #     end_time - start_time
        # )

        # max_memory = (
        #     torch.cuda.max_memory_allocated() / 1024.0**3
        #     if torch.cuda.is_available()
        #     else None
        # )

        # results = {
        #     "input_tokens": input_tokens,
        #     "output_tokens": output_tokens,
        #     "tokens_per_second": tokens_per_second,
        #     "elapsed_time": end_time - start_time,
        #     "max_cuda_memory": max_memory,
        #     "output_text": output_text,
        # }

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
            output = model(text, **kwargs)
            end_time = time.time()

            input_tokens = output["usage"]["prompt_tokens"]
            output_tokens = output["usage"]["completion_tokens"]
            output_text = output["choices"][0]

            tokens_per_second = (output_tokens) / (
                end_time - start_time
            )

            max_memory = (
                torch.cuda.max_memory_allocated() / 1024.0**3
                if torch.cuda.is_available()
                else None
            )

            results["input_tokens"].append(input_tokens)
            results["total_tokens"].append(output_tokens)
            results["tokens_per_second"].append(tokens_per_second)
            results["elapsed_time"].append(end_time - start_time)
            results["max_cuda_memory"].append(max_memory)
            results["output_text"].append(output_text)
    # cleanup
    return results



def profile_generate_text(input_text, model, tokenizer, **kwargs):
    if isinstance(input_text, str):
        input_text = [input_text]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_tensors=True,
        **kwargs,
    )

    return None


def clear_model(model, tokenizer):
    del model
    del tokenizer
    del pipeline
    torch.cuda.empty_cache()

    return None


def torch_profile_to_dataframe(prof):
    """
    Convert a TorchProfiler output to a pandas DataFrame.

    Args:
        prof (torch.autograd.profiler.profile): TorchProfiler output.

    Returns:
        pandas.DataFrame: DataFrame containing the profiled data.

    Raises:
        None
    """
    prof_output = prof.key_averages().table(sort_by="self_cuda_time_total")
    lines = [
        line for line in prof_output.split("\n") if not line.startswith("-")
    ]

    # Convert the processed string back to a single string
    processed_str = "\n".join(
        lines[:-3]
    )  # omitting the last two lines with totals

    # Convert the string to a DataFrame
    df = pd.read_csv(StringIO(processed_str), sep="\s\s+", engine="python")

    # Handle percent columns
    percent_cols = ["Self CPU %", "CPU total %", "Self CUDA %"]
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].str.rstrip("%").astype("float") / 100.0

    # Handle time columns
    time_cols = [
        "Self CPU",
        "CPU total",
        "CPU time avg",
        "Self CUDA",
        "CUDA total",
        "CUDA time avg",
    ]
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].apply(_parse_time)

    return df


def _parse_time(time_str):
    # Parse time values with unit suffixes (e.g., ms, us, s) to microseconds
    time_str = time_str.strip()
    if time_str.endswith("us"):
        return float(time_str.rstrip("us"))
    elif time_str.endswith("ms"):
        return float(time_str.rstrip("ms")) * 1e3
    elif time_str.endswith("s"):
        return float(time_str.rstrip("s")) * 1e6
    else:
        return float(time_str)
