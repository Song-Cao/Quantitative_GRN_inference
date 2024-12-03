import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import networkx as nx

def infer_trajectory(adata, method="dpt", pseudotime_column="pseudotime"):
    """
    Infer trajectory using diffusion pseudotime or other methods.

    Args:
        adata (AnnData): Preprocessed single-cell data.
        method (str): Method for trajectory inference ('dpt', 'slingshot').
        pseudotime_column (str): Column name to store pseudotime information.
    """
    if method == "dpt":
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
        adata.obs[pseudotime_column] = adata.obs["dpt_pseudotime"]
    elif method == "slingshot":
        try:
            import slingshot
            slingshot.tl.slingshot(adata)
            adata.obs[pseudotime_column] = adata.obs["slingshot_pseudotime"]
        except ImportError:
            raise ImportError("Slingshot is not installed. Install it with 'pip install slingshot'")
    else:
        raise ValueError(f"Unsupported trajectory inference method: {method}")
    return adata

def initialize_grn(adata, method="coexpression", threshold=0.5):
    """
    Initialize a Gene Regulatory Network (GRN).

    Args:
        adata (AnnData): Single-cell data.
        method (str): Method for GRN initialization ('coexpression', 'priors').
        threshold (float): Threshold for co-expression similarity.

    Returns:
        adjacency_matrix (np.ndarray): GRN adjacency matrix.
    """
    print("Initializing GRN...")
    if method == "coexpression":
        # Compute gene-gene co-expression similarity
        gene_expression = adata.X.T  # Genes as rows
        correlation_matrix = np.corrcoef(gene_expression)
        adjacency_matrix = (correlation_matrix > threshold).astype(int)
    elif method == "priors":
        raise NotImplementedError("GRN initialization with priors is not yet implemented.")
    else:
        raise ValueError(f"Unsupported GRN initialization method: {method}")

    return adjacency_matrix

def construct_dataset(adata, adjacency_matrix, output_dir, test_size=0.2, val_size=0.1, stratify_column=None):
    """
    Split data into training, validation, and testing sets.

    Args:
        adata (AnnData): Annotated data with pseudotime.
        adjacency_matrix (np.ndarray): GRN adjacency matrix.
        output_dir (str): Directory to save datasets.
        test_size (float): Fraction of data for testing.
        val_size (float): Fraction of data for validation.
        stratify_column (str): Column in .obs to stratify splits.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = adata.to_df()
    labels = adata.obs["pseudotime"]
    stratify = adata.obs[stratify_column] if stratify_column else None

    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=(test_size + val_size), stratify=stratify, random_state=42)
    val_size_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, stratify=stratify, random_state=42)

    X_train.to_csv(f"{output_dir}/train_data.csv", index=False)
    X_val.to_csv(f"{output_dir}/val_data.csv", index=False)
    X_test.to_csv(f"{output_dir}/test_data.csv", index=False)
    y_train.to_csv(f"{output_dir}/train_labels.csv", index=False)
    y_val.to_csv(f"{output_dir}/val_labels.csv", index=False)
    y_test.to_csv(f"{output_dir}/test_labels.csv", index=False)
    
    # Save GRN adjacency matrix
    np.save(f"{output_dir}/grn_adjacency_matrix.npy", adjacency_matrix)
    print(f"Datasets and GRN saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Infer trajectory and construct dataset")
    parser.add_argument("--input", required=True, help="Path to input h5ad file")
    parser.add_argument("--output_dir", required=True, help="Output directory for datasets")
    parser.add_argument("--method", choices=["dpt", "slingshot"], default="dpt", help="Trajectory inference method")
    parser.add_argument("--grn_method", choices=["coexpression", "priors"], default="coexpression", help="Method for GRN initialization")
    parser.add_argument("--grn_threshold", type=float, default=0.5, help="Threshold for GRN coexpression")
    parser.add_argument("--stratify_column", type=str, help="Column for stratified splitting")
    args = parser.parse_args()

    adata = sc.read_h5ad(args.input)
    adata = infer_trajectory(adata, method=args.method)
    grn_adjacency_matrix = initialize_grn(adata, method=args.grn_method, threshold=args.grn_threshold)
    construct_dataset(adata, grn_adjacency_matrix, args.output_dir, stratify_column=args.stratify_column)
