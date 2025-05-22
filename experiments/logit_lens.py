# %% redoing the logit lens experiment... sadly can't find original data
import json
import torch as t
import os
from nnsight import LanguageModel
from transformers import AutoTokenizer

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
base_steering_vectors = t.load("data/steering_vectors/steering_base_vectors_offset12_window4.pt")['backtracking'].to("cuda")
ft_steering_vectors = t.load("data/steering_vectors/steering_vectors_offset12_window4.pt")['backtracking'].to("cuda")
base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
ft_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# Also load the models
base_model = LanguageModel("meta-llama/Llama-3.1-8B", device_map="cuda", torch_dtype=t.bfloat16)
ft_model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", torch_dtype=t.bfloat16)

# %%
from tqdm import tqdm
logit_lens_results = t.zeros(base_steering_vectors.shape[0], 4, device="cuda")
# b2b, b2f, f2b, f2f for the shape[1] dimension
# just to actually load the models instead of meta...
with base_model.trace([0]) as tr:
    pass

with ft_model.trace([0]) as tr:
    pass




# do topk decoding for each logit
k = 15


vocab_set = ["wait"]
# decode the topk tokens
for layer in tqdm(range(base_model.config.num_hidden_layers)):
    # base trained steering vectors, decoded by the base model
    logits = (base_model.lm_head.weight.data.to(t.float32) @ base_steering_vectors.T).T
    topk_vals, topk_inds = t.topk(logits, k=k, dim=1)
    for i in range(k):
        token = base_tokenizer.decode(topk_inds[layer, i])
        if any(v in token.strip().lower() for v in vocab_set):
            logit_lens_results[layer, 0] += topk_vals[layer, i]
            
    # base trained steering vectors, decoded by the ft model
    logits = (ft_model.lm_head.weight.data.to(t.float32) @ base_steering_vectors.T).T
    topk_vals, topk_inds = t.topk(logits, k=k, dim=1)
    for i in range(k):
        token = ft_tokenizer.decode(topk_inds[layer, i])
        if any(v in token.strip().lower() for v in vocab_set):
            logit_lens_results[layer, 1] += topk_vals[layer, i]
    
    # ft trained steering vectors, decoded by the base model
    logits = (base_model.lm_head.weight.data.to(t.float32) @ ft_steering_vectors.T).T
    topk_vals, topk_inds = t.topk(logits, k=k, dim=1)
    for i in range(k):
        token = base_tokenizer.decode(topk_inds[layer, i])
        if any(v in token.strip().lower() for v in vocab_set):
            logit_lens_results[layer, 2] += topk_vals[layer, i]
    
    # ft trained steering vectors, decoded by the ft model
    logits = (ft_model.lm_head.weight.data.to(t.float32) @ ft_steering_vectors.T).T
    topk_vals, topk_inds = t.topk(logits, k=k, dim=1)
    for i in range(k):
        token = ft_tokenizer.decode(topk_inds[layer, i])
        if any(v in token.strip().lower() for v in vocab_set):
            logit_lens_results[layer, 3] += topk_vals[layer, i]
        

# %% visualization
import plotly.express as px
import plotly.graph_objects as go

# results = logit_lens_results.cpu().numpy()
# fig = go.Figure()

# # Add traces for each column
# fig.add_trace(go.Scatter(y=results[:, 0], name='Base Steering - Base Model'))
# fig.add_trace(go.Scatter(y=results[:, 1], name='Base Steering - FT Model'))
# fig.add_trace(go.Scatter(y=results[:, 2], name='FT Steering - Base Model'))
# fig.add_trace(go.Scatter(y=results[:, 3], name='FT Steering - FT Model'))

# # Update layout
# fig.update_layout(
#     title='Logit Lens Results Across Layers',
#     xaxis_title='Layer',
#     yaxis_title='Logit Value',
#     showlegend=True
# )

import matplotlib.pyplot as plt

results = logit_lens_results.cpu().numpy()
plt.figure(figsize=(10, 6))

# Plot lines with same color but different opacity for same decoder
plt.plot(results[:, 0], label='Base-derived steering vectors; Base model decoded', color='tab:blue', alpha=0.5, marker='o', linestyle='-')
plt.plot(results[:, 1], label='Base-derived steering vectors; FT model decoded', color='tab:blue', alpha=1.0, marker='o', linestyle='-')
plt.plot(results[:, 2], label='FT-derived steering vectors; Base model decoded', color='tab:orange', alpha=0.5, marker='o', linestyle='-')
plt.plot(results[:, 3], label='FT-derived steering vectors; FT model decoded', color='tab:orange', alpha=1.0, marker='o', linestyle='-')

# plt.title('Logit Lens Results Across Layers', fontsize=14, pad=15)
plt.xlabel('Layer', fontsize=15, labelpad=10)
plt.ylabel('Backtracking score', fontsize=15, labelpad=10)
plt.legend(fontsize=13)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Adjust layout to prevent text cutoff
plt.tight_layout()

plt.savefig("logit_lens_results.png", dpi=300, bbox_inches='tight')



# %%