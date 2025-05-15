# %% 

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Set CUDA launch blocking for better error messages
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import torch as t
from nnsight import LanguageModel, apply
from transformers import AutoTokenizer
# from datasets import load_dataset
import pickle
import html
import pandas as pd
import sys
from typing import Union
# import plotly.express as px
# import plotly.graph_objects as go
from torch.distributions import Categorical
# from plotly.subplots import make_subplots
# from memory_util import MemoryMonitor
import gc
# Other stuff
import math
import re
from IPython.display import HTML, display
from tqdm import tqdm
import json
# from datasets import load_dataset
import torch.nn.functional as F
from src.annotate_reasoning_chains import load_reasoning_chains
from openai import OpenAI

# %% Loading the model and tokenizer

# pick a question from the reasoning tasks dataset
# Also need the steering vector outcome

from src.generate_reasoning_chains import load_model_and_tokenizer, generate_reasoning_chain

model, tokenizer = load_model_and_tokenizer()
# model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", torch_dtype=t.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# %% Also load the reasoning tasks dataset

task_set = json.load(open("data/tasks/all_reasoning_tasks.json", "r"))
task_example = task_set[0]
question = task_example["problem"]
# %%
reasoning_chain = generate_reasoning_chain(model, tokenizer, question)

# analyze the reasoning chain based on the llm judge or the manual judge
# %%
from src.annotate_reasoning_chains import annotate_chain

annotated_chain = annotate_chain(reasoning_chain)


# %% access steering vectors

def generate_steered_chains(
    nns_model: LanguageModel,     # fine-tuned model
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
        with nns_model.generate(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
        ) as tracer:
            activation = nns_model.model.layers[steering_layer].output[0]
            activation[:] += steering_magnitude * steering_vector.to(activation.device)
            
            steered_rollout = nns_model.generator.output.save()
    
    steered_chain = tokenizer.decode(steered_rollout[0])
    return steered_chain, formatted_prompt


# %%
STEERING_VECTORS_DIR = "data/steering_vectors"
base_steering_vectors = t.load(os.path.join(STEERING_VECTORS_DIR, "base_steering_vectors.pt"))
finetune_steering_vectors = t.load(os.path.join(STEERING_VECTORS_DIR, "ft_steering_vectors.pt"))
steering_vector = base_steering_vectors["backtracking"]
steering_layer = 10
steering_magnitude = 50

steered_chain, formatted_prompt = generate_steered_chains(
    nns_model,
    tokenizer,
    question,
    steering_vector,
    steering_layer,
    steering_magnitude,
    max_new_tokens=500
)

# %%
from IPython.display import HTML, display
display(HTML(steered_chain))

# %%

# comparing the annotations



