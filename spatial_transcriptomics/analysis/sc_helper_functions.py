import anndata
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.sparse import issparse, csr_matrix

def process_gene(gene_name, datasets_paths, exp, saving_folder, cluster_name, category, n):
    exp_cluster = []
    for dsn, adata_path in datasets_paths.items():
        adata = anndata.read_h5ad(adata_path, backed='r')
        gene_mask = adata.var["gene_symbol"] == gene_name
        exp_cluster.append(adata[:, gene_mask])

    combined_exp_cluster = anndata.concat(exp_cluster, axis=0)
    exp_values = combined_exp_cluster.to_df().values.flatten()
    exp_df = exp[['x', 'y']]
    sorted_exp_df = exp_df.reindex(combined_exp_cluster.obs.index)
    sorted_exp_df.reset_index(inplace=True)

    # Sort the cells by expression level for the max projection effect
    sorted_indices = exp_values.argsort()
    sorted_x = sorted_exp_df['x'].values[sorted_indices]
    sorted_y = sorted_exp_df['y'].values[sorted_indices]
    sorted_exp_values = exp_values[sorted_indices]

    # Plotting
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    scatter = ax.scatter(sorted_x, sorted_y, s=0.1, c=sorted_exp_values, marker='.', cmap=plt.cm.magma_r)

    ax.axis('equal')
    ax.set_xlim(-18, 27)
    ax.set_ylim(-18, 27)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{gene_name}")

    # Create a colorbar with the expression values
    # plt.colorbar(scatter, ax=ax, label='Expression Level')

    # Save the figure
    plt.savefig(os.path.join(saving_folder, f"10x_gene_{n}_{gene_name}_{cluster_name}_{category}.png"), dpi=300)
    plt.close(fig)


def min_max_normalize(adata):
    # Check if the data is stored as a sparse matrix
    if issparse(adata.X):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X

    # Calculate the minimum and maximum values for each gene
    X_min = X_dense.min(axis=0)
    X_max = X_dense.max(axis=0)

    # Find where the range is zero (to avoid division by zero)
    range_zero = X_max == X_min
    # Set the range to 1 for these cases to avoid division by zero (it will become zero later with nan_to_num)
    ranges = np.where(range_zero, 1, X_max - X_min)

    # Perform min-max normalization
    X_norm = (X_dense - X_min) / ranges

    # Replace any NaNs and Infs resulting from division by zero with zeros
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # Create a new AnnData object with the normalized data, if the original is sparse, convert back to sparse format
    if issparse(adata.X):
        X_norm_sparse = csr_matrix(X_norm)
        adata_norm = anndata.AnnData(X_norm_sparse, obs=adata.obs, var=adata.var, dtype='float32')
    else:
        adata_norm = anndata.AnnData(X_norm, obs=adata.obs, var=adata.var, dtype='float32')

    return adata_norm