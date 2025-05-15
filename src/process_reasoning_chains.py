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


VOCAB_SENTENCE_ENDINGS = [
    ".", "!", "?", ".\n", ".\n\n", ". ", "? ", "! ", ".\n ", ".\n\n "
]
REASONING_VOCAB = {
    "restatement": ["Okay", "okay", "Ok", "let me", "Let me", "Now", "now", "Let's", "let's"],
    "deduction": ["So", "so", "therefore", "thus", "Thus", "Therefore", "Hence", "hence", "perhaps", "Perhaps", "Similarly", "similarly"],
    "backtracking": ["But", "but", "Wait", "wait", "However", "however", "Alternatively", "alternatively"],
    "conclusion": ["correct", "work", "right", "yes", "yeah", "Yes", "Yeah", "that's all", "That's all", "that is all", "That is all"]
}
COLOR_DICT = {
    "restatement": "red",
    "deduction": "green",
    "backtracking": "blue",
    "conclusion": "purple",
    "other": "gray"
}



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

def separate_sentences(
    trace: str,
    tokenizer: AutoTokenizer,
    vocab_sentence_endings: list[str] = VOCAB_SENTENCE_ENDINGS,
    print_msg: bool = True,
) -> tuple[list[str], list[str]]:
    """Convert a reasoning trace into separate sentences.

    This function takes a reasoning trace string and splits it into individual sentences
    based on sentence-ending punctuation. It handles both regular sentences and bullet points
    by filtering out very short sequences.

    Args:
        trace (str): The reasoning trace text to split into sentences
        tokenizer (AutoTokenizer): Tokenizer for encoding/decoding text
        vocab_sentence_endings (list[str], optional): List of sentence ending markers. 
            Defaults to VOCAB_SENTENCE_ENDINGS.
        print_msg (bool, optional): Whether to print colored sentence visualization. 
            Defaults to True.

    Returns:
        tuple[list[str], list[list[int]]]: A tuple containing:
            - sentence_strs: List of sentence strings, each list element is a string
            - sentence_tokens: List of tokenized sentences, each list element is a list of integers
    """

    msg_tokens = tokenizer.encode(trace)
    sentence_break_inds = []
    for i, token in enumerate(msg_tokens):
        if any(ending in tokenizer.decode(token) for ending in vocab_sentence_endings):
            # Manually ruling out the case where "." is actually a decimal point, like in "1.414"
            is_decimal = False
            try:
                number_str = tokenizer.decode(
                    [msg_tokens[i-1], msg_tokens[i], msg_tokens[i+1]],
                )
                strs = number_str.split(".")
                if strs[0][-1].isdigit() and strs[1][0].isdigit():
                    is_decimal = True
            except:
                pass
        
            if not is_decimal:
                sentence_break_inds.append(i)
        
    # To deal with bullet points, need to filter out the sentences with length <= 3
    filtered_sentence_break_inds = [0]
    active_sentence_start = 0
    for i, ind in enumerate([0] + sentence_break_inds):
        if i == 0:
            continue
        
        # sentence_length = ind - active_sentence_start
        # # print(sentence_length)
        # if sentence_length <= 3:
        #     continue
        # else:
        filtered_sentence_break_inds.append(ind)
        active_sentence_start = ind
            
    
    # Now break the message into sentences
    # NOTE: The operation here has missed the first BOS token
    sentence_tokens = [
        msg_tokens[filtered_sentence_break_inds[i]+1:filtered_sentence_break_inds[i+1]+1]
        for i in range(len(filtered_sentence_break_inds)-1)
    ]
    sentence_strs = [
        tokenizer.decode(sentence_tokens[i])
        for i in range(len(sentence_tokens))
    ]
    
    if print_msg:
        html = ""
        color = ["red", "green"]
        for i, s in enumerate(sentence_strs):
            html += f"<span style='color: {color[i % 2]}'>{s}</span>"
            
        display(HTML(html))
    return sentence_strs, sentence_tokens, filtered_sentence_break_inds


def classify_sentences(
    sentence_strs: list[str],
    sentence_tokens: list[list[int]],
    reasoning_vocab: dict[str, list[str]] = REASONING_VOCAB,
    return_dict: bool = False,
    print_msg: bool = True,
    color_dict: dict[str, str] = COLOR_DICT,
) -> Union[list[str], list[dict[str, float]]]:
    """
    Classify sentences into reasoning categories based on vocabulary patterns.
    
    Args:
        sentence_strs (list[str]): List of sentence strings to classify
        sentence_tokens (list[list[int]]): List of tokenized sentences
        reasoning_vocab (dict[str, list[str]], optional): Dictionary mapping reasoning categories to keyword lists. Defaults to REASONING_VOCAB.
        return_dict (bool, optional): If True, returns detailed scores for each category. If False, returns only the highest scoring category. Defaults to False.
        print_msg (bool, optional): If True, displays colored HTML output of classified sentences. Defaults to True.
        color_dict (dict[str, str], optional): Dictionary mapping categories to colors for HTML display. Defaults to COLOR_DICT.
    
    Returns:
        Union[list[str], list[dict[str, float]]]: If return_dict is False, returns list of category names. If True, returns list of dictionaries containing scores for each category.
    
    Note:
        - Scores are weighted based on keyword position (beginning vs end of sentence)
        - Capitalized keywords receive higher weights
        - Conclusion words at sentence endings receive extra weight
        - "other" category has a default score of 0.6
    """
    
    
    scores = {
        "restatement": 0,
        "deduction": 0, 
        "backtracking": 0,
        "conclusion": 0,
        "other": 0.6
    }
    cats = []
    for tkn, st in zip(sentence_tokens, sentence_strs):
        for category, keywords in reasoning_vocab.items():
            sentence_beginning = st[:min(len(st), 14)] # Note this is first 14 string elements, not tokens
            sentence_else = st[-min(len(st), 14):]
            for keyword in keywords:
                if keyword in sentence_beginning:
                    if keyword[0].isupper():
                        scores[category] += 1.0
                    else:
                        scores[category] += 0.5
                        
                if keyword in sentence_else:
                    if keyword[0].isupper():
                        scores[category] += 0.5
                    else:
                        scores[category] += 0.25
                    
        # Add extra weight for conclusion words at end of sentence
        for keyword in reasoning_vocab["conclusion"]:
            if st.strip().lower().endswith(keyword.lower()+"."):
                scores["conclusion"] += 1.5
        
        if return_dict:
            cats.append(scores)
        else:
            cats.append(max(scores, key=scores.get))
        
        # reset the values of scores
        scores = {key: 0 if key != "other" else 0.6 for key in scores.keys()}
    
    if print_msg:
        if return_dict:
            raise NotImplementedError("Printing with return_dict=True not implemented, please use return_dict=False instead")

        html = ""
        for i, s in enumerate(sentence_strs):
            html += f"<span style='color: {color_dict[cats[i]]}'>{s}</span>"
            
        display(HTML(html))

    return cats


def process_annotated_chain(annotated_chain: str):
    """Process the annotated chains to get the sentence strings and the corresponding categories
    
    Note that this is from the LLM judge's perspective.
    
    Args:
        annotated_chain (str): The annotated chain to process
        
    """
    
    # split the reasoning chain into sentences
    text_chunks = annotated_chain.split('["end-section"]')
    cats = []
    strs = []
    for chunk in text_chunks:
        if chunk == '':
            continue
        try:
            text_cat, text_str = chunk.split(']')
            text_cat = text_cat.strip('\n').strip('["').strip('"')
            cats.append(text_cat)
            strs.append(text_str)
        except:
            print(f"Error splitting chunk: {chunk}")
            raise Exception("Error splitting chunk")
    return strs, cats