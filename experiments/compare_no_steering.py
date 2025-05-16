import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dotenv
import re

dotenv.load_dotenv()
print("Starting directory:", os.getcwd())
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
print("Changed to directory:", os.getcwd())
import dotenv
dotenv.load_dotenv()
from src.generate_steered_traces import sample_steering_trace
from src.generate_reasoning_chains import load_model_and_tokenizer
from src.process_reasoning_chains import process_annotated_chain, separate_sentences, classify_sentences
from src.annotate_reasoning_chains import annotate_chain
from src.generate_steered_traces import sample_steering_trace
from nnsight import LanguageModel
import torch as t
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

def compare_labellings_no_steering(
    reasoning_chain_data: dict,
):
    annotated_chain = reasoning_chain_data["annotated_chain"]
    reasoning_chain = reasoning_chain_data["reasoning_chain"]
    
    strs_llm, cats_llm = process_annotated_chain(annotated_chain)
    strs_manual, sentence_tokens_manual, filtered_sentence_break_inds_manual = separate_sentences(reasoning_chain, tokenizer, print_msg=False)
    cats_manual = classify_sentences(strs_manual, sentence_tokens_manual,print_msg=False)
    
    btk_llm = [i for i, cat in zip(strs_llm, cats_llm) if cat == "backtracking"]
    btk_manual = [i for i, cat in zip(strs_manual, cats_manual) if cat == "backtracking"]
    btk_llm = [btk_llm[i].strip() for i in range(len(btk_llm))]
    btk_manual = [btk_manual[i].strip() for i in range(len(btk_manual))]
    
    is_sub_element = lambda s, set: any(s in sub for sub in set) or any(sub in s for sub in set)
    # is_sub_element = lambda s, set: any(s == sub for sub in set)
    only_in_llm = [s for s in btk_llm if not is_sub_element(s, btk_manual)]
    only_in_manual = [s for s in btk_manual if not is_sub_element(s, btk_llm)]
    
    return only_in_llm, only_in_manual, btk_llm, btk_manual

# reasoning_chain_data = json.load(open("data/annotated_chains/all_annotated_chains.json", "r"))[12]
# only_in_llm, only_in_manual, btk_llm, btk_manual = compare_labellings_no_steering(reasoning_chain_data)

def gather_comparison_stats(
    file_path: str,
):
    counts = {
        "only_in_llm": 0,
        "only_in_manual": 0,
        "llm_positives": 0,
        "kw_positives": 0,
    }
    reasoning_chain_data = json.load(open(file_path, "r"))
    for chain in tqdm(reasoning_chain_data):
        try:
            only_in_llm, only_in_manual, btk_llm, btk_manual = compare_labellings_no_steering(chain)
        except Exception as e:
            print(f"Error on chain {chain['task_id']}: {e}, skipping...")
            continue
    
        counts["only_in_llm"] += len(only_in_llm)
        counts["only_in_manual"] += len(only_in_manual)
        counts["llm_positives"] += len(btk_llm)
        counts["kw_positives"] += len(btk_manual)
      
    return counts

def compare_llm_and_keyword_labelling(annotation_file_path: list[str] | str, keywords: list[str] = ["wait"]):
    if isinstance(annotation_file_path, str):
        annotation_file_path = [annotation_file_path]
    
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total": 0}
    for file_path in annotation_file_path:
        steered_reasoning_chain_data = json.load(open(file_path, "r"))
        for chain_data in steered_reasoning_chain_data:
            results = [0, 0] # represents [llm, keyword], 0 for negative, 1 for positive
            annotated_chain = chain_data["annotated_chain"]
            strs_llm, cats_llm = process_annotated_chain(annotated_chain)
            for (sentence, cat) in zip(strs_llm, cats_llm):
                if cat == "backtracking":
                    results[0] = 1
                if re.search(r"\bwait\b", sentence, re.IGNORECASE):
                    results[1] = 1
            
            if results[0] == 1 and results[1] == 1:
                counts["TP"] += 1
            elif results[0] == 0 and results[1] == 1:
                counts["FP"] += 1
            elif results[0] == 1 and results[1] == 0:
                counts["FN"] += 1
            elif results[0] == 0 and results[1] == 0:
                counts["TN"] += 1
            
            counts["total"] += 1
    
    return counts
                

if __name__ == "__main__":
    counts = gather_comparison_stats("data/annotated_chains/all_annotated_chains.json")
    print(counts)
    