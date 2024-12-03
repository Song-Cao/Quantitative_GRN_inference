# Quantitative GRN Inference from Spatio-temporal Single-Cell Data

This repository implements a pipeline to infer dynamic Gene Regulatory Networks (GRNs) from spatial and temporal single-cell data using an integrated Graph Attention Network (GAT) and Transformer architecture.

## Features
- **Data Preprocessing**: Quality control and normalization of spatial single-cell data.
- **Trajectory and GRN Inference**: Pseudotime trajectories inference and regulatory graph initialization
- **Integrated GAT-Transformer Model**: Combines local interpretability of GATs with global dependency modeling of Transformers.
- **End-to-End Pipeline**: From data preprocessing to model training and evaluation.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Song-Cao/Quantitative_GRN_Inference.git
    cd Quantitative_GRN_Inference
    ```

2. Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```