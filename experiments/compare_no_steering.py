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
# from src.generate_reasoning_chains import load_model_and_tokenizer
from src.process_reasoning_chains import process_annotated_chain, separate_sentences, classify_sentences
# from src.annotate_reasoning_chains import annotate_chain

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

def compare_llm_and_keyword_labelling(annotation_file_path: list[str] | str, keywords: list[str] = ["wait"], save_to_path: str = None):
    if isinstance(annotation_file_path, str):
        annotation_file_path = [annotation_file_path]
    
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total_sentence_analyzed": 0, "skipped_instances": 0, "metadata": annotation_file_path}
    for file_path in annotation_file_path:
        try:
            # explicitly handling two formats of recorded steering traces
            steered_reasoning_chain_data = json.load(open(file_path, "r"))["traces"]
        except:
            steered_reasoning_chain_data = json.load(open(file_path, "r"))
        
        # print(steered_reasoning_chain_data)
        for data_index, chain_data in enumerate(steered_reasoning_chain_data):
            # print(chain_data)
            try:
                strs_llm, cats_llm = process_annotated_chain(chain_data["annotated_chain"])
            except Exception as e:
                print(f"Error on chain {data_index} of {file_path}, skipping...")
                counts["skipped_instances"] += 1
                continue
            for (sentence, cat) in zip(strs_llm, cats_llm):
                results = [0, 0] # represents [llm, keyword], 0 for negative, 1 for positive
                if cat == "backtracking":
                    results[0] = 1
                if any([re.search(rf"\b{keyword}\b", sentence, re.IGNORECASE) for keyword in keywords]):
                # if re.search(r"\bwait\b", sentence, re.IGNORECASE):
                    results[1] = 1
            
                if results[0] == 1 and results[1] == 1:
                    counts["TP"] += 1
                elif results[0] == 0 and results[1] == 1:
                    counts["FP"] += 1
                elif results[0] == 1 and results[1] == 0:
                    counts["FN"] += 1
                elif results[0] == 0 and results[1] == 0:
                    counts["TN"] += 1
            
                counts["total_sentence_analyzed"] += 1
    
    if save_to_path is not None:
        assert save_to_path.endswith('.json'), f"File {save_to_path} must be a JSON file"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
        
        # Initialize with empty dict if file doesn't exist or is empty
        if not os.path.exists(save_to_path) or os.path.getsize(save_to_path) == 0:
            data = []
        else:
            with open(save_to_path, "r") as f:
                data = json.load(f)
        
        # Update data and write back
        data.append(counts)
        with open(save_to_path, "w") as f:
            json.dump(data, f, indent=4)

    return counts


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    # file_path = "data/annotated_chains/all_annotated_chains.json"
    
    for magnitude in [4, 8, 12]:
        file_dir = f"data/new_steering_results_finetune/magnitude_{magnitude}"
        file_path = os.listdir(file_dir)
        file_path = [os.path.join(file_dir, file) for file in file_path]
        # print(file_path)
        # file_path = "data/steering_results/magnitude_4/probability_annotated_chains.json"
        save_to_path = f"data/new_steering_results_finetune/counts.json"
        keywords = ["wait", "alternatively"]
        counts = compare_llm_and_keyword_labelling(file_path, keywords=keywords, save_to_path=save_to_path)
        print(counts)

if __name__ == "__main__":
    main()