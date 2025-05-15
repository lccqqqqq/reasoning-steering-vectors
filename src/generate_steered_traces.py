import json
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import pandas as pd
from typing import Dict
from nnsight import LanguageModel
from transformers import AutoTokenizer
import os
import pickle
import itertools
from tqdm import tqdm

# from run_experiment import load_random_prompts
# from generate_steering_vectors import convert_to_base_tokens

def load_random_prompts(file_path, num_prompts):
    with open(file_path, "r") as f:
        all_prompts = json.load(f)
    
    return random.sample(all_prompts, num_prompts)


def convert_to_base_tokens(tokens: t.Tensor):
    """
    Convert r1 tokens to base tokens. Only works for Llama tokenizers.
    """
    # patch_token = 77627 # ` ############`
    patch_token = 27370 # ` ####`
    tokens = tokens.clone()
    tokens[tokens == 128011] = patch_token
    tokens[tokens == 128012] = patch_token
    tokens[tokens == 128013] = patch_token
    tokens[tokens == 128014] = patch_token
    return tokens


# setup the model
base_model = LanguageModel("meta-llama/Llama-3.1-8B", device_map="cuda", torch_dtype=t.bfloat16)
finetune_model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", torch_dtype=t.bfloat16)

finetune_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# load the steering vectors
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
STEERING_VECTORS_DIR = "data/steering_vectors"
base_steering_vectors = t.load(os.path.join(STEERING_VECTORS_DIR, "base_steering_vectors.pt"))
finetune_steering_vectors = t.load(os.path.join(STEERING_VECTORS_DIR, "ft_steering_vectors.pt"))

def generate_steered_traces(
    model: LanguageModel,
    base_tokenizer: AutoTokenizer,
    finetune_tokenizer: AutoTokenizer,
    model_type: str,
    prompts: list[Dict[str, str]],
    steering_vectors: Dict[str, t.Tensor],
    steering_category: str,
    steering_layer: int,
    max_new_tokens: int = 128,
    num_traces_per_steer: int = 5,
    steering_magnitude: float = 1.0,
    save_to_path: str = None,
):
    
    # formatted_prompts = []
    total_skipped = 0
    # making a dataframe to store the traces
    traces = pd.DataFrame(columns=["prompt", "original_trace", "simulated_trace", "steered_trace"])
    for prompt in prompts:
        # Create a new row as a dictionary and append it to the DataFrame
        new_row = {
            "prompt": prompt["problem"],
            "original_trace": prompt["reasoning_chain"]
        }
        # Use pandas concat or loc to add the row
        
        
        
        
        if model_type == "base":
            # patch the start of the reasoning trace to the base model to prevent early-stage repetitions.
            # need to check for consistency of the prompt through the tokenizer
            
            # patch the start of the reasoning trace to the base model to prevent early-stage repetitions.
            # ensure that the model generation is initally on track...
            reasoning_chain_base = base_tokenizer.encode(prompt["reasoning_chain"])
            starter_tokens = reasoning_chain_base[:128]
            start_of_reasoning_chain = base_tokenizer.decode(starter_tokens)
            
            formatted_prompt = (
                prompt["problem"] + "\n" + start_of_reasoning_chain
            )
            base_tokens = convert_to_base_tokens(base_tokenizer.encode(formatted_prompt, add_special_tokens=False, return_tensors="pt")[0])
            base_text = base_tokenizer.decode(base_tokens)
            base_tokens2 = base_tokenizer.encode(base_text, add_special_tokens=False, return_tensors="pt")[0]
            if not t.equal(base_tokens, base_tokens2):
                print(f"tokenizer mismatch, skipping...")
                total_skipped += 1
                # traces["simulated_trace"].append(None)
                # traces["steered_trace"].append(None)
                continue

        elif model_type == "finetune":
            # do we need to check for consistency (?)
            formatted_prompt = finetune_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["problem"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        
        # So now the consistency check has passed, either model should output the same tokens
        tokenizer = finetune_tokenizer if model_type == "finetune" else base_tokenizer
        # generate with the steering vector
        steered_traces = []
        original_traces = []
        for _ in range(num_traces_per_steer):
            with t.no_grad():
                with model.generate(
                    formatted_prompt,
                    max_new_tokens=max_new_tokens,
                ) as tracer:
                    activation = model.model.layers[steering_layer].output[0]
                    activation[:] += steering_magnitude * steering_vectors[steering_category][steering_layer].to(activation.device)
                    
                    steered_rollout = model.generator.output.save()
            
            steered_traces.append(tokenizer.decode(steered_rollout[0]))
            
            with t.no_grad():
                with model.generate(
                    formatted_prompt,
                    max_new_tokens=max_new_tokens,
                ) as tracer:
                    original_rollout = model.generator.output.save()
            
            original_traces.append(tokenizer.decode(original_rollout[0]))
        
        new_row["simulated_trace"] = original_traces
        new_row["steered_trace"] = steered_traces
        
        traces = pd.concat([traces, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save traces to a JSON file
    if save_to_path is not None:
        # Create directory if it doesn't exist
        # os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
        
        # Format the data for saving
        save_data = {
            "metadata": {
                "model_type": model_type,
                "steering_category": steering_category,
                "steering_layer": steering_layer,
                "steering_magnitude": steering_magnitude,
                "num_traces_per_steer": num_traces_per_steer,
                "max_new_tokens": max_new_tokens,
                "total_skipped": total_skipped
            },
            "traces": traces.to_dict(orient="records")
        }
        
        # Save to JSON file
        with open(save_to_path, 'w') as f:
            json.dump(save_data, f, indent=4)
        
        print(f"Saved traces to {save_to_path}")
    
    return traces


def collect_traces_all_categories(
    model: LanguageModel,
    base_tokenizer: AutoTokenizer,
    finetune_tokenizer: AutoTokenizer,
    model_type: str,
    prompts: list[Dict[str, str]],
    steering_vectors: Dict[str, t.Tensor],
    save_dir: str,
):
    # setting up data collector, for all categories
    categories = list(steering_vectors.keys())[1:]
    traces = []
    steering_magnitudes = [32, -32]
    steering_layers = [11, 5]
    max_new_tokens = 256
    repetition = 4
    save_paths_data = itertools.product(categories, steering_magnitudes, steering_layers)
    os.makedirs(save_dir, exist_ok=True)
    for category, steering_magnitude, steering_layer in tqdm(save_paths_data):
        save_path = os.path.join(save_dir, f"steering_traces_{category}_magnitude_{steering_magnitude}_layer_{steering_layer}.json")
        trace = generate_steered_traces(
            model=model,
            base_tokenizer=base_tokenizer,
            finetune_tokenizer=finetune_tokenizer,
            model_type=model_type,
            prompts=prompts,
            steering_vectors=steering_vectors,
            steering_category=category,
            steering_layer=steering_layer,
            steering_magnitude=steering_magnitude,
            max_new_tokens=max_new_tokens,
            num_traces_per_steer=repetition,
            save_to_path=save_path,
        )
        traces.append(trace)
        
    return traces


def sample_steering_trace(
    model: LanguageModel,     # fine-tuned model
    tokenizer: AutoTokenizer, # fine-tuned tokenizer
    question: str,
    steering_vector: t.Tensor,
    steering_layer: int,
    steering_magnitude: float,
    max_new_tokens: int = 500
):
    # Generate steered chains given a steering vector, on the fine-tuned model's rollout from a given prompt
    
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(formatted_prompt)
    with t.no_grad():
        with model.generate(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
        ) as tracer:
            activation = model.model.layers[steering_layer].output[0]
            activation[:] += steering_magnitude * steering_vector.to(activation.device)
            
            steered_rollout = model.generator.output.save()
    
    steered_chain = tokenizer.decode(steered_rollout[0], skip_special_tokens=True)
    return steered_chain, formatted_prompt


if __name__ == "__main__":
    prompts = load_random_prompts("data/annotated_chains/all_annotated_chains.json", 12)
    traces = collect_traces_all_categories(
        model=base_model,
        base_tokenizer=base_tokenizer,
        finetune_tokenizer=finetune_tokenizer,
        model_type="base",
        prompts=prompts,
        steering_vectors=base_steering_vectors,
        save_dir="steered_rollouts_large_magnitude",
    )
    # traces = generate_steered_traces(
    #     model=base_model,
    #     tokenizer=base_tokenizer,
    #     model_type="base",
    #     prompts=prompts,
    #     steering_vectors=base_steering_vectors,
    #     steering_category="initializing",
    #     steering_layer=10,
    #     max_new_tokens=128,
    #     num_traces_per_steer=1,
    #     steering_magnitude=10,
    #     save_to_path="base_steering_traces_test.json",
    # )



