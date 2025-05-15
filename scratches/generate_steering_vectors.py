import json
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Union

def load_annotated_chain(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_annotations(annotated_text):
    annotations = []
    
    current_pos = 0
    while True:
        start_tag_pos = annotated_text.find('[\"', current_pos)
        if start_tag_pos == -1:
            break
            
        end_tag_pos = annotated_text.find('\"]', start_tag_pos)
        if end_tag_pos == -1:
            break
            
        category = annotated_text[start_tag_pos+2:end_tag_pos]
        
        if category == "end-section":
            current_pos = end_tag_pos + 2
            continue
            
        start_text_pos = end_tag_pos + 2
        end_section_tag = annotated_text.find('[\"end-section\"]', start_text_pos)
        
        if end_section_tag != -1:
            text = annotated_text[start_text_pos:end_section_tag].strip()
            annotations.append((category, text))
            current_pos = end_section_tag + 15
        else:
            current_pos = end_tag_pos + 2
    
    return annotations

def process_chain(
    model_or_tokenizer: Union[LanguageModel, AutoTokenizer],
    chain: Dict
):
    problem = chain["problem"]
    # supporting input types of both LanguageModel and AutoTokenizer
    tokenizer = model_or_tokenizer.tokenizer if isinstance(model_or_tokenizer, LanguageModel) else model_or_tokenizer
    
    formatted_problem = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    tokenized_problem = tokenizer.encode(formatted_problem, return_tensors="pt", add_special_tokens=False)[0]
    
    annotations = extract_annotations(chain["annotated_chain"])
    
    annotation_indices = {}
    
    current_token_pos = len(tokenized_problem)
    
    full_tokens = tokenized_problem.tolist()
    
    for i, (category, text) in enumerate(annotations):
        if i > 0:
            text = " " + text
        segment_tokens = tokenizer.encode(text, add_special_tokens=False)
        
        start_idx = current_token_pos
        end_idx = start_idx + len(segment_tokens) - 1
        
        if category not in annotation_indices:
            annotation_indices[category] = []
        annotation_indices[category].append((start_idx, end_idx))
        
        full_tokens.extend(segment_tokens)
        
        current_token_pos += len(segment_tokens)
    
    tokenized_full_text = torch.tensor(full_tokens)
    
    return tokenized_full_text, annotation_indices

def process_chains_iterator(
    model_or_tokenizer: Union[LanguageModel, AutoTokenizer],
    chains: List[Dict]
):
    for i, chain in enumerate(chains):
        tokenized_text, annotation_indices = process_chain(model_or_tokenizer, chain)
        
        chain_id = chain.get('task_id', f'chain_{i}')
        yield tokenized_text, annotation_indices

def generate_steering_vectors(file_path, layer_of_interest=10, offset=-20, window_size=4):
    ## annotations...
    chains = load_annotated_chain(file_path)
    model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", torch_dtype=torch.bfloat16)
    ## Averaging over all reasoning chains in the dataset
    processed_chains = process_chains_iterator(model, chains)
    
    activations = defaultdict(list)
    skipped_examples = defaultdict(int)

    with torch.inference_mode():
        for tokens, indices in tqdm(processed_chains):
            text = model.tokenizer.decode(tokens)
            tokens2 = model.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
            assert torch.equal(tokens, tokens2)
            with model.trace(text) as tracer:
                layer_activations = model.model.layers[layer_of_interest].output[0].save()
            torch.cuda.empty_cache()
            for category, index_tuples in indices.items():
                if category not in ["backtracking", "uncertainty-estimation", "initializing", "deduction", "example-testing", "adding-knowledge"]:
                    continue
                for start, end in index_tuples:
                    # Apply offset to start index and set window
                    adjusted_start = start + offset
                    adjusted_end = adjusted_start + window_size
                    
                    # Check if the adjusted indices are valid
                    if adjusted_start < 1:  # 1-indexed, so start must be at least 1
                        skipped_examples[category] += 1
                        continue
                    
                    # Collect activations for the adjusted window
                    activations[category].append(layer_activations[0, adjusted_start-1:adjusted_end].float().cpu())
    
    # Print skipped examples if there are any
    total_skipped = sum(skipped_examples.values())
    if total_skipped > 0:
        print(f"Skipped {total_skipped} examples because they would go beyond the beginning:")
        for category, count in skipped_examples.items():
            print(f"  - {category}: {count} examples skipped")
    
    # Calculate overall mean
    overall_mean = torch.zeros(4096, dtype=torch.float64)
    num_activations = 0
    for _, layer_activations in activations.items():
        for la in layer_activations:
            overall_mean += la.to(torch.float64).sum(dim=0)
            num_activations += la.shape[0]
    overall_mean /= num_activations
    overall_mean = overall_mean.to(torch.float32)
    
    # Calculate mean vectors for each category
    mean_vectors = {}
    for category, layer_activations in activations.items():
        category_mean = torch.zeros(4096, dtype=torch.float64)
        num_activations = 0
        for la in layer_activations:
            category_mean += la.to(torch.float64).sum(dim=0)
            num_activations += la.shape[0]
        category_mean /= num_activations
        mean_vectors[category] = category_mean.to(torch.float32)
    
    # Compute steering vectors as difference-of-means
    steering_vectors = {}
    for category, mean_vector in mean_vectors.items():
        steering_vectors[category] = mean_vectors[category] - overall_mean
    
    # Normalize steering vectors
    steering_vectors = {k: v / v.norm() for k, v in steering_vectors.items()}
    
    return steering_vectors, overall_mean, model


def convert_to_base_tokens(tokens: torch.Tensor):
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


def generate_base_steering_vectors(file_path, layer_of_interest=10, offset=-20, window_size=4):
    model = LanguageModel("meta-llama/Llama-3.1-8B", device_map="cuda", torch_dtype=torch.bfloat16)
    # subtleties with base tokenizers
    ## annotations...
    chains = load_annotated_chain(file_path)
    ft_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    processed_chains = process_chains_iterator(ft_tokenizer, chains)
    
    activations = defaultdict(list)
    skipped_examples = defaultdict(int)
    num_skipped = 0


    with torch.inference_mode():
        for ft_tokens, indices in tqdm(processed_chains):
            text = ft_tokenizer.decode(ft_tokens)
            ft_tokens2 = ft_tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
            assert torch.equal(ft_tokens, ft_tokens2)
            
            # To base tokens
            base_tokens = convert_to_base_tokens(ft_tokens)
            base_text = base_tokenizer.decode(base_tokens)
            base_tokens2 = base_tokenizer.encode(base_text, add_special_tokens=False, return_tensors="pt")[0]
            if not torch.equal(base_tokens, base_tokens2):
                print(f"tokenizer mismatch, skipping...")
                num_skipped += 1
                continue
            
            with model.trace(text) as tracer:
                layer_activations = model.model.layers[layer_of_interest].output[0].save()

            torch.cuda.empty_cache()
            
            for category, index_tuples in indices.items():
                if category not in ["backtracking", "uncertainty-estimation", "initializing", "deduction", "example-testing", "adding-knowledge"]:
                    print(f"Found spurious category: {category}, skipping...")
                    num_skipped += 1
                    continue
                for start, end in index_tuples:
                    # Apply offset to start index and set window
                    adjusted_start = start + offset
                    adjusted_end = adjusted_start + window_size
                    
                    # Check if the adjusted indices are valid
                    if adjusted_start < 1:  # 1-indexed, so start must be at least 1
                        skipped_examples[category] += 1
                        num_skipped += 1
                        continue
                    
                    # Collect activations for the adjusted window
                    activations[category].append(layer_activations[0, adjusted_start-1:adjusted_end].float().cpu())
    
    # Print skipped examples if there are any
    total_skipped = sum(skipped_examples.values())
    if total_skipped > 0:
        print(f"Skipped {total_skipped} examples because they would go beyond the beginning:")
        for category, count in skipped_examples.items():
            print(f"  - {category}: {count} examples skipped")

    print(f"Skipped {num_skipped}/{len(chains)} examples in total")
    
    # Calculate overall mean
    # need to convert to float64 as otherwise the sum of bfloat16 will overflow
    overall_mean = torch.zeros(4096, dtype=torch.float64)
    num_activations = 0
    for _, layer_activations in activations.items():
        for la in layer_activations:
            overall_mean += la.to(torch.float64).sum(dim=0)
            num_activations += la.shape[0]
    overall_mean /= num_activations
    overall_mean = overall_mean.to(torch.float32)
    
    # Calculate mean vectors for each category
    mean_vectors = {}
    for category, layer_activations in activations.items():
        category_mean = torch.zeros(4096, dtype=torch.float64)
        num_activations = 0
        for la in layer_activations:
            category_mean += la.to(torch.float64).sum(dim=0)
            num_activations += la.shape[0]
        category_mean /= num_activations
        mean_vectors[category] = category_mean.to(torch.float32)
    
    # Compute steering vectors as difference-of-means
    steering_vectors = {}
    for category, mean_vector in mean_vectors.items():
        steering_vectors[category] = mean_vectors[category] - overall_mean
    
    # Normalize steering vectors
    steering_vectors = {k: v / v.norm() for k, v in steering_vectors.items()}
    
    return steering_vectors, overall_mean, model
    

def generate_steering_vectors_with_offset(file_path, layer_of_interest=10, offset=0, window_size=4):
    """
    Convenience function for backward compatibility
    """
    return generate_steering_vectors(file_path, layer_of_interest, offset, window_size)

if __name__ == "__main__":
    # steering_vectors, overall_mean, model = generate_steering_vectors("annotated_chains/all_annotated_chains.json")
    steering_vectors, overall_mean, model = generate_base_steering_vectors("annotated_chains/all_annotated_chains.json")
    print("Generated steering vectors for categories:", list(steering_vectors.keys())) 