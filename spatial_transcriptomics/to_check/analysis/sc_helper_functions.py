import os

import numpy as np
import anndata
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix

import utils.utils as ut

NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]

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
    plt.savefig(os.path.join(saving_folder, f"10x_gene_{n}_{gene_name}_{cluster_name}_{category}.svg"), dpi=300)
    plt.close(fig)


def plot_gene_expression_in_10X_data(gene_name, datasets_paths, exp, saving_folder, saving_name="", mask_name="",
                                     cluster_name=""):
    # Create the directory for saving the output
    saving_directory = ut.create_dir(os.path.join(saving_folder, gene_name))

    # Load the expression data for the specified gene from all datasets
    exp_cluster = []
    for dsn, adata_path in datasets_paths.items():
        adata = anndata.read_h5ad(adata_path, backed='r')
        gene_mask = adata.var["gene_symbol"] == gene_name
        exp_cluster.append(adata[:, gene_mask])

    # Concatenate expression data across datasets
    combined_exp_cluster = anndata.concat(exp_cluster, axis=0)
    exp_values = combined_exp_cluster.to_df().values.flatten()
    exp_values = [2**i for i in exp_values]
    np.save(os.path.join(saving_directory, f"min_max_{gene_name}.npy"), np.array([np.min(exp_values), np.max(exp_values)]))

    # NORMALIZE
    ut.print_c(f"[INFO] Min/Max expression: {np.min(exp_values)}/{np.max(exp_values)}")
    # exp_values = (exp_values - np.min(exp_values)) / (np.max(exp_values) - np.min(exp_values))
    exp_values = np.array(exp_values)
    exp_values[exp_values >= 1500] = 1500
    exp_values = (exp_values - 0) / (1500 - 0)

    # Prepare the dataframe containing cell coordinates
    exp_df = exp[['x', 'y', 'class', 'cluster']]
    sorted_exp_df = exp_df.reindex(combined_exp_cluster.obs.index)
    sorted_exp_df.reset_index(inplace=True)

    if mask_name == "neurons":
        mask = np.array([False if any([j in str(i) for j in NON_NEURONAL_CELL_TYPES]) else True
                         for i in sorted_exp_df["class"]])
    elif mask_name == "non_neurons":
        mask = np.array([True if any([j in str(i) for j in NON_NEURONAL_CELL_TYPES]) else False
                         for i in sorted_exp_df["class"]])
    if cluster_name:
        mask = np.array([False if str(i) == cluster_name else True
                         for i in sorted_exp_df["cluster"]])

    if mask_name or cluster_name:
        # Ensure mask is a flattened array corresponding to the same indices as combined_exp_cluster
        masked_cells_x = sorted_exp_df['x'][mask]
        masked_cells_y = sorted_exp_df['y'][mask]

        non_masked_cells_x = sorted_exp_df['x'][~mask]
        non_masked_cells_y = sorted_exp_df['y'][~mask]
        non_masked_exp_values = exp_values[~mask]

        # Sort non-masked cells by expression values
        sorted_indices_non_masked = non_masked_exp_values.argsort()
        # sorted_x_non_masked = non_masked_cells_x.values[sorted_indices_non_masked]
        sorted_x_non_masked = non_masked_cells_x
        # sorted_y_non_masked = non_masked_cells_y.values[sorted_indices_non_masked]
        sorted_y_non_masked = non_masked_cells_y
        # sorted_exp_values_non_masked = non_masked_exp_values[sorted_indices_non_masked]
        sorted_exp_values_non_masked = non_masked_exp_values
    else:
        # Sort all cells by expression values if no mask is provided
        sorted_indices = exp_values.argsort()
        sorted_x_non_masked = sorted_exp_df['x'].values[sorted_indices]
        sorted_y_non_masked = sorted_exp_df['y'].values[sorted_indices]
        sorted_exp_values_non_masked = exp_values[sorted_indices]

    # Create the figure and axis
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)

    if mask_name or cluster_name:
        # Plot masked cells in gray
        ax.scatter(masked_cells_x, masked_cells_y, s=0.05, c='#efefef', marker='.')

    # Plot non-masked cells with the expression colormap
    scatter = ax.scatter(sorted_x_non_masked, sorted_y_non_masked, s=0.05,
                         c=sorted_exp_values_non_masked, marker='.', cmap=plt.cm.gist_heat_r, vmin=0, vmax=1)  #YlOrRd

    ax.axis('equal')
    ax.set_xlim(-18, 27)
    ax.set_ylim(-18, 27)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{gene_name}")

    # Save the figure
    plt.savefig(os.path.join(saving_directory, f"10x_gene_{gene_name}{saving_name}.png"), dpi=300)
    plt.savefig(os.path.join(saving_directory, f"10x_gene_{gene_name}{saving_name}.svg"), dpi=300)
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
