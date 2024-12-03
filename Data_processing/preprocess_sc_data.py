import scanpy as sc
import pandas as pd
import numpy as np
import os

def preprocess_spatial_data(
    input_file, 
    output_file, 
    min_genes=200, 
    min_cells=3, 
    mito_gene_prefix="MT-", 
    scale_data=True, 
    n_pcs=50, 
    batch_correction=False, 
    batch_key=None
):
    """
    Preprocessing of spatial single-cell data.

    Args:
        input_file (str): Path to the input file in h5ad format.
        output_file (str): Path to save the preprocessed data.
        min_genes (int): Minimum number of genes for a cell to be included.
        min_cells (int): Minimum number of cells for a gene to be included.
        mito_gene_prefix (str): Prefix for mitochondrial genes.
        scale_data (bool): Whether to scale data.
        n_pcs (int): Number of principal components for dimensionality reduction.
        batch_correction (bool): Apply batch correction if True.
        batch_key (str): Key in .obs for batch correction.
    """
    # Load data
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")
    adata = sc.read_h5ad(input_file)
    
    # Quality control: Filter cells and genes
    print("Performing quality control...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Annotate mitochondrial gene content
    print("Annotating mitochondrial genes...")
    adata.var["mt"] = adata.var_names.str.startswith(mito_gene_prefix)
    adata.obs["percent_mito"] = np.sum(
        adata[:, adata.var["mt"]].X, axis=1
    ) / np.sum(adata.X, axis=1)
    
    # Filter cells with high mitochondrial content
    adata = adata[adata.obs["percent_mito"] < 0.2, :]
    
    print("Normalizing data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Identify highly variable genes
    print("Identifying highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    # Batch effect correction
    if batch_correction and batch_key:
        print("Performing batch correction...")
        if batch_key not in adata.obs:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs.")
        sc.pp.combat(adata, key=batch_key)

    # Scale data and perform PCA
    if scale_data:
        print("Scaling data and performing PCA...")
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=n_pcs)

    # Save preprocessed data
    print(f"Saving preprocessed data to {output_file}...")
    adata.write_h5ad(output_file)
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess spatial single-cell data")
    parser.add_argument("--input", required=True, help="Path to input file in h5ad format")
    parser.add_argument("--output", required=True, help="Path to output file in h5ad format")
    parser.add_argument("--min_genes", type=int, default=200, help="Minimum number of genes per cell")
    parser.add_argument("--min_cells", type=int, default=3, help="Minimum number of cells per gene")
    parser.add_argument("--mito_prefix", type=str, default="MT-", help="Prefix for mitochondrial genes")
    parser.add_argument("--batch_key", type=str, help="Key for batch correction")
    args = parser.parse_args()
    
    preprocess_spatial_data(
        input_file=args.input,
        output_file=args.output,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
        mito_gene_prefix=args.mito_prefix,
        batch_correction=bool(args.batch_key),
        batch_key=args.batch_key,
    )
