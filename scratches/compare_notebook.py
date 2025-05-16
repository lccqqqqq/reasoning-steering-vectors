# %%
import os

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import dotenv
dotenv.load_dotenv()
from src.generate_steered_traces import sample_steering_trace
from src.generate_reasoning_chains import load_model_and_tokenizer
from nnsight import LanguageModel
import torch as t
import json
model, tokenizer = load_model_and_tokenizer()
nns_model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", torch_dtype=t.bfloat16)
# %% Loading steering vectors

STEERING_VECTORS_DIR = "data/steering_vectors"
base_steering_vector = t.load(os.path.join(STEERING_VECTORS_DIR, "base_steering_vectors.pt"))['backtracking']
finetune_steering_vector = t.load(os.path.join(STEERING_VECTORS_DIR, "ft_steering_vectors.pt"))['backtracking']

# other iterables
task_set = json.load(open("data/tasks/all_reasoning_tasks.json", "r"))
task_example = task_set[15]
question = task_example["problem"]
# print(question)
steering_strength_set = [0.0, 1.0, 4.0, 8.0, 10.0, 15.0, 30.0]
steering_layer = 10

# %% Test generation and annotation

from src.process_reasoning_chains import process_annotated_chain, separate_sentences, classify_sentences
from src.annotate_reasoning_chains import annotate_chain
from src.generate_steered_traces import sample_steering_trace

steering_magnitude = 3.0
steered_chain, formatted_prompt = sample_steering_trace(
    nns_model,
    tokenizer,
    question,
    finetune_steering_vector,
    steering_layer,
    steering_magnitude,
    max_new_tokens=300
)

# analyze by llm judge
llm_annotated_chain = annotate_chain(steered_chain)
strs_llm, cats_llm = process_annotated_chain(llm_annotated_chain)
# %%
# analyze by manual judge

steered_chain_manual = tokenizer.decode(tokenizer.encode(steered_chain, add_special_tokens=False)[len(tokenizer.encode(formatted_prompt, add_special_tokens=False))-1:])
strs_manual, sentence_tokens_manual, filtered_sentence_break_inds_manual = separate_sentences(steered_chain_manual, tokenizer, print_msg=False)
cats_manual = classify_sentences(strs_manual, sentence_tokens_manual,print_msg=False)

# %% brief test
# Now need to assert that the two chains have the same number of sentences
print(len(strs_llm))
print(len(strs_manual))
print(strs_llm)
print(strs_manual)
print(cats_llm)
print(cats_manual)


btk_llm = [i for i, cat in zip(strs_llm, cats_llm) if cat == "backtracking"]
btk_manual = [i for i, cat in zip(strs_manual, cats_manual) if cat == "backtracking"]

print(btk_llm)
print(btk_manual)

# %%

def compare_labellings(
    task_problem: str,
    steering_vector: t.Tensor,
    steering_magnitude: float,
    steering_layer: int,
    max_new_tokens: int = 300,
):
    steered_chain, formatted_prompt = sample_steering_trace(
        nns_model,
        tokenizer,
        task_problem,
        steering_vector,
        steering_layer,
        steering_magnitude,
        max_new_tokens=max_new_tokens
    )

    llm_annotated_chain = annotate_chain(steered_chain)
    strs_llm, cats_llm = process_annotated_chain(llm_annotated_chain)

    steered_chain_manual = tokenizer.decode(tokenizer.encode(steered_chain, add_special_tokens=False)[len(tokenizer.encode(formatted_prompt, add_special_tokens=False))-1:])
    strs_manual, sentence_tokens_manual, filtered_sentence_break_inds_manual = separate_sentences(steered_chain_manual, tokenizer, print_msg=False)
    cats_manual = classify_sentences(strs_manual, sentence_tokens_manual,print_msg=False)

    btk_llm = [i for i, cat in zip(strs_llm, cats_llm) if cat == "backtracking"]
    btk_manual = [i for i, cat in zip(strs_manual, cats_manual) if cat == "backtracking"]
    btk_llm = [btk_llm[i].strip() for i in range(len(btk_llm))]
    btk_manual = [btk_manual[i].strip() for i in range(len(btk_manual))]

    # systematically compare the backtracking sentences
    # Find strings in btk_llm but not in btk_manual
    is_sub_element = lambda s, set: any(s in sub for sub in set) or any(sub in s for sub in set)
    only_in_llm = [s for s in btk_llm if not is_sub_element(s, btk_manual)]
    # Find strings in btk_manual but not in btk_llm
    only_in_manual = [s for s in btk_manual if not is_sub_element(s, btk_llm)]
    
    print(f"Number of backtracking sentences only in LLM annotation: {len(only_in_llm)}/{len(btk_llm)}")
    print(f"Number of backtracking sentences only in manual annotation: {len(only_in_manual)}/{len(btk_manual)}")
    
    if len(only_in_llm) > 0:
        print("\nSentences only in LLM annotation:")
        for s in only_in_llm:
            print(f"- {s}")
            
    if len(only_in_manual) > 0:
        print("\nSentences only in manual annotation:")
        for s in only_in_manual:
            print(f"- {s}")

    return only_in_llm, only_in_manual, btk_llm, btk_manual, steered_chain_manual
# %%
from tqdm import tqdm
# only_in_llm, only_in_manual, btk_llm, btk_manual = compare_labellings(task_example["problem"], base_steering_vector, 5.0, 10)

# need to sweep through reasoning chains 
def sweep(file_path: str, steering_magnitude_set: list[float], steering_vector: t.Tensor, steering_layer: int = 10, save_dir: str = 'data/steering_results', steering_type: str = "base", save_file: bool = True, max_new_tokens: int = 300):
    task_set = json.load(open(file_path, "r"))
    os.makedirs(save_dir, exist_ok=True)
    for steering_magnitude in steering_magnitude_set:
        print(f"Sweeping through steering magnitude {steering_magnitude}")
        save_path = os.path.join(save_dir, steering_type + f"_{int(steering_magnitude)}")
        os.makedirs(save_path, exist_ok=True)
        counts = {
            "only_in_llm": 0,
            "only_in_manual": 0,
            "btk_llm": 0,
            "btk_manual": 0,
        }
        task_simulation_results = []
        for task in tqdm(task_set):
            try:
                only_in_llm, only_in_manual, btk_llm, btk_manual, steered_chain_manual = compare_labellings(task["problem"], steering_vector, steering_magnitude, steering_layer, max_new_tokens)
            except Exception as e:
                print(f"Error on task {task['id']}: {e}, skipping...")
                continue
            counts["only_in_llm"] += len(only_in_llm)
            counts["only_in_manual"] += len(only_in_manual)
            counts["btk_llm"] += len(btk_llm)
            counts["btk_manual"] += len(btk_manual)
            task_simulation_results.append({
                "task_id": task["id"],
                "task_problem": task["problem"],
                "task_answer": steered_chain_manual,
                "only_in_llm": only_in_llm,
                "only_in_manual": only_in_manual,
                "btk_llm": btk_llm,
                "btk_manual": btk_manual,
            })
        
        if save_file:
            with open(os.path.join(save_path, "counts.json"), "w") as f:
                json.dump(counts, f)
            
            with open(os.path.join(save_path, "task_simulation_results.json"), "w") as f:
                json.dump(task_simulation_results, f)
        

# %%
sweep("data/tasks/all_reasoning_tasks.json", [1.0], base_steering_vector, 10, save_dir="data/steering_results/base", steering_type="base", save_file=True, max_new_tokens=500)





