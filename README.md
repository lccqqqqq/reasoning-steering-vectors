# LLM Reasoning Behavior Analysis

This repository contains code and experiments for analyzing reasoning behavior in Large Language Models, with a particular focus on steering vectors and backtracking mechanisms.

## Project Overview

This research investigates how thinking models perform sophisticated operations and learn complex behaviors, particularly focusing on:
- The role of steering vectors in triggering backtracking behavior
- The relationship between steering vectors and uncertainty metrics
- The systematic nature of backtracking in response to high vector projections

## Project Structure

```
reasoning_steering_vectors/
├── data/                      # Data storage and processing
│   ├── raw/                   # Raw experimental data
│   └── processed/             # Processed datasets
├── experiments/               # Individual experiment modules
│   ├── steering_vector/       # Core steering vector analysis
│   ├── backtracking/          # Backtracking behavior analysis
│   ├── uncertainty/           # Uncertainty metrics comparison
│   └── semantic_entropy/      # Semantic entropy analysis
├── src/                       # Source code
│   ├── models/               # Model interfaces and utilities
│   ├── metrics/              # Evaluation metrics
│   └── utils/                # Helper functions
├── notebooks/                # Jupyter notebooks for analysis
├── configs/                  # Configuration files
└── results/                  # Experimental results and visualizations
```

## Key Experiments

1. **Steering Vector Analysis**
   - Vector activation patterns
   - Input/output token analysis
   - Max activation examples
   - Trigger timing analysis

2. **Backtracking Verification**
   - Causal relationship verification
   - Natural text analysis
   - Projection-based prediction

3. **Uncertainty Metrics Comparison**
   - Weight unembedding comparison
   - Entropy neurons analysis
   - Per-token entropy analysis
   - Semantic entropy evaluation

4. **LLM-based Evaluation**
   - Robust sentence analysis
   - Weight metric validation

## Setup and Installation

[Installation instructions to be added]

## Usage

[Usage instructions to be added]

## Citation

[Citation information to be added] 