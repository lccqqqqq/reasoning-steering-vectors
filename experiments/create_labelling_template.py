# Never thought I'd do this one day...

# From reasoning chains, create two files of reasoning traces
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from src.process_reasoning_chains import process_annotated_chain

# Constants
FILEPATHS = [
    "data/new_steering_results/magnitude_4/math_logic_annotated_chains.json",
    "data/new_steering_results/magnitude_8/probability_annotated_chains.json",
    "data/new_steering_results/magnitude_4/spatial_annotated_chains.json",
]
SAVE_PATH = "data/labelling_reasoning_traces"
os.makedirs(SAVE_PATH, exist_ok=True)
TEMPLATE_FILENAME = "labelling_template.json"
KEYWORD_JUDGE_FILENAME = "keyword_judge_labelled_reasoning_chunks.json"

keywords = ["wait", "alternatively"]

unlabelled_traces = []
labelled_traces = []
index = 0
for filepath in FILEPATHS:
    data = json.load(open(filepath))
    traces = data["traces"]
    print(f"Processing {filepath}")
    for trace in traces:
        unlabelled_reasoning_chunks = []
        labelled_reasoning_chunks = []
        annotated_chain = trace["annotated_chain"]
        try:
            strs, cats = process_annotated_chain(annotated_chain)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
        
        for (sentence, cat) in zip(strs, cats):
            # use keyword judge
            if any([re.search(rf"\b{keyword}\b", sentence, re.IGNORECASE) for keyword in keywords]):
                labelled_reasoning_chunks.append({sentence: 1})
            else:
                labelled_reasoning_chunks.append({sentence: 0})
            
            unlabelled_reasoning_chunks.append({sentence: 0})
    
        unlabelled_traces.append({
            "id": index,
            "prompt": trace["prompt"],
            "reasoning_chunks": unlabelled_reasoning_chunks,
        })
        labelled_traces.append({
            "id": index,
            "prompt": trace["prompt"],
            "reasoning_chunks": labelled_reasoning_chunks,
        })
        index += 1
    
# save the traces
with open(os.path.join(SAVE_PATH, TEMPLATE_FILENAME), "w") as f:
    json.dump(unlabelled_traces, f, indent=4)
    
print(f"Saved {len(unlabelled_traces)} unlabelled traces to {os.path.join(SAVE_PATH, TEMPLATE_FILENAME)}")

with open(os.path.join(SAVE_PATH, KEYWORD_JUDGE_FILENAME), "w") as f:
    json.dump(labelled_traces, f, indent=4)


print(f"Saved {len(labelled_traces)} labelled traces to {os.path.join(SAVE_PATH, KEYWORD_JUDGE_FILENAME)}")
            
            