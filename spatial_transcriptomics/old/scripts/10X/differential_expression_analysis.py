import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import scanpy as sc
import requests

# Use the Agg backend to avoid rendering plots interactively
import matplotlib

matplotlib.use("Agg")  # Ensure backend is non-interactive for speed

clusters = pd.read_excel(r"E:\tto\spatial_transcriptomics_results\CVOs\results\3d_views\counts_cells_cluster.xlsx")
clusters_names = clusters["Label"]
# Selected cluster names (all will be plotted together)
defined_cluster_names = np.array(clusters_names)
defined_cluster_names = np.append(defined_cluster_names, "3864 SNc-VTA-RAmb Foxa1 Dopa_4")

category_type = "cluster"

saving_dir = r"E:\tto\spatial_transcriptomics_results\CVOs\results"
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

dataset_id = "WMB-10X"  # Select the dataset
download_base = r'E:\tto\spatial_transcriptomics'  # Path to data on the local drive

# Manifest url
url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
manifest = json.loads(requests.get(url).text)  # Load the manifest
metadata_json = manifest['file_listing'][dataset_id]['metadata']  # Fetch metadata structure

# Path to metadata file (updated to the correct structure)
metadata_relative_path = metadata_json['cell_metadata_with_cluster_annotation']['files']['csv']['relative_path']
metadata_file = os.path.join(download_base, metadata_relative_path)  # Path to metadata file
exp = pd.read_csv(metadata_file, low_memory=False)  # Load metadata
exp.set_index('cell_label', inplace=True)  # Set cell_label as dataframe index
clusters = exp["cluster"]
unique_clusters = np.unique(clusters)
if category_type == "supertype":
    supertypes = exp["supertype"]
    defined_cluster_names = np.unique(clusters[supertypes == defined_cluster_names[0]])
else:
    defined_cluster_names = defined_cluster_names

metadata_genes_relative_path = metadata_json['gene']['files']['csv']['relative_path']
metadata_gene_file = os.path.join(download_base, metadata_genes_relative_path)  # Path to metadata file
genes = pd.read_csv(metadata_gene_file)  # Load metadata

# Save the results
output_dir = os.path.join(saving_dir, "scanpy_results")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for n, cluster_names in enumerate([defined_cluster_names]):

    # if n == 0:
    #     prefix = "global"
    if n == 0:
        prefix = "defined"

    # Initialize a list to store the data from all clusters
    all_cluster_data = {}
    cell_cluster_mapping = []

    # Add total count trackers
    total_clusters = len(cluster_names)

    for cluster_idx, cluster_name in enumerate(cluster_names):
        print(f"Processing cluster {cluster_idx + 1}/{total_clusters}: {cluster_name}")

        # Ensure that the selected cluster is fetched as before
        cluster_mask = exp["cluster"] == cluster_name  # Mask of the cells belonging to the selected cluster
        library_mask = exp["library_method"] == "10Xv3"  # Mask of the library data (optional)
        cluster_and_library_mask = np.logical_and(cluster_mask, library_mask)

        # Get the cells and gene expression data for the selected cluster
        selected_cells = exp[cluster_and_library_mask].index
        adata_cluster_in = []  # Gene expression of cells in the selected cluster for each sub-dataset

        print(f"Found {len(selected_cells)} cells in {cluster_name}")

        # Dataset name for expression matrices
        dataset_id = "WMB-10Xv3"
        metadata_exp = manifest['file_listing'][dataset_id]['expression_matrices']

        # Load the gene expression data for the selected cluster
        for dsn in metadata_exp:
            adata = anndata.read_h5ad(
                os.path.join(download_base, metadata_exp[dsn]["log2"]["files"]["h5ad"]["relative_path"]), backed='r')
            mask_in = adata.obs.index.isin(selected_cells)
            adata_cluster_in.append(adata[mask_in])

        # Combine all the datasets for this cluster
        adata_cluster_in_filtered = [x for x in adata_cluster_in if len(x) > 0]  # Filter out empty data
        combined_adata_in = anndata.concat(adata_cluster_in_filtered, axis=0)  # Combine datasets by rows (cells)

        # Append the combined gene expression data for this cluster to the list
        all_cluster_data[cluster_name] = combined_adata_in.to_df()  # Add the data to a dictionary for each cluster

        # Keep track of the cluster each cell belongs to (for the fragmented bar)
        cell_cluster_mapping.extend([cluster_name] * combined_adata_in.shape[0])

    # Step 1: Combine all cluster data into a single dataframe
    combined_data_all_clusters = pd.concat(all_cluster_data, axis=0)

    # Step 2: Add a "cluster" column to keep track of which cells belong to which cluster
    combined_data_all_clusters['cluster'] = cell_cluster_mapping

    # Step 3: Convert the combined dataframe into an AnnData object
    # Dropping the 'cluster' column as it's categorical and not part of expression matrix
    adata = anndata.AnnData(X=combined_data_all_clusters.drop(columns=['cluster']).values)
    adata.obs['cluster'] = combined_data_all_clusters['cluster'].values  # Add cluster info to obs
    adata.var['gene_names'] = combined_data_all_clusters.drop(columns=['cluster']).columns  # Add gene names to var

    # Step 4: Normalize and log-transform the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Step 5: Perform PCA to reduce dimensions before differential expression
    sc.tl.pca(adata)

    # Step 6: Perform differential expression analysis using PCA-reduced data
    # Specify that you want to use the PCA representation to avoid the warning
    sc.tl.rank_genes_groups(adata, groupby='cluster', method='wilcoxon', use_rep='X_pca')

    # Step 7: Visualize the top genes per cluster (optional)
    sc.pl.rank_genes_groups_heatmap(
        adata,
        n_genes=5,
        groupby='cluster',
        show=False,
        cmap='coolwarm'  # Specify the colormap here
    )

    plt.savefig(os.path.join(output_dir, f"{prefix}_diff_expr_heatmap_top5.png"))
    plt.savefig(os.path.join(output_dir, f"{prefix}_diff_expr_heatmap_top5.svg"))

    # Step 4: Visualize the top genes per cluster (optional)
    sc.pl.rank_genes_groups_heatmap(
        adata,
        n_genes=50,
        groupby='cluster',
        show=False,
        cmap='coolwarm'  # Specify the colormap here
    )

    plt.savefig(os.path.join(output_dir, f"{prefix}_diff_expr_heatmap_top50.png"))
    plt.savefig(os.path.join(output_dir, f"{prefix}_diff_expr_heatmap_top50.svg"))

########################################################################################################################
# SAVE THE RESULT IN CSV
########################################################################################################################

# Load the gene metadata to map gene identifiers to gene symbols
metadata_genes_relative_path = metadata_json['gene']['files']['csv']['relative_path']
metadata_gene_file = os.path.join(download_base, metadata_genes_relative_path)  # Path to metadata file
genes_metadata = pd.read_csv(metadata_gene_file)  # Load metadata

# Ensure the genes_metadata DataFrame has the relevant columns
genes_metadata = genes_metadata.rename(columns={'gene_identifier': 'gene_id', 'gene_symbol': 'gene_acronym'})

# Extract the differential expression results from Scanpy
clusters = adata.uns['rank_genes_groups']['names'].dtype.names

genes_identifiers = adata.var

# Loop through each cluster and save results separately
for cluster in clusters:
    # Get the gene indices for the current cluster
    gene_indices = adata.uns['rank_genes_groups']['names'][cluster].astype(int)

    # Map the gene indices to gene names
    gene_names = [str(genes_identifiers.iloc[i].values[0]) for i in gene_indices]

    # Get the statistics for the current cluster
    pvals = adata.uns['rank_genes_groups']['pvals'][cluster]
    logfoldchanges = adata.uns['rank_genes_groups']['logfoldchanges'][cluster]
    scores = adata.uns['rank_genes_groups']['scores'][cluster]

    # Combine the data into a DataFrame for the current cluster
    cluster_results = pd.DataFrame({
        'gene_id': gene_names,  # This column will contain the gene identifiers
        'p_value': pvals,
        'logfoldchange': logfoldchanges,
        'score': scores
    })

    # Step 1: Map the gene IDs to gene acronyms (symbols) from the metadata
    cluster_results_with_acronyms = cluster_results.merge(genes_metadata[['gene_id', 'gene_acronym']], on='gene_id',
                                                          how='left')

    # Step 2: Save the DataFrame for the current cluster
    cluster_csv_path = os.path.join(output_dir, f"differentially_expressed_genes_{cluster}.csv")
    cluster_results_with_acronyms.to_csv(cluster_csv_path, index=False)

    print(f"Results for cluster {cluster} saved to: {cluster_csv_path}")

########################################################################################################################
# CROSS-CORRELATION
########################################################################################################################

# Step 1: Compute mean expression for each cluster
mean_expression_per_cluster = adata.to_df().groupby(adata.obs['cluster']).mean()

# Step 2: Compute the correlation matrix across clusters
correlation_matrix = mean_expression_per_cluster.T.corr()

# Step 3: Sort the correlation matrix by cluster names (optional)
correlation_matrix = correlation_matrix.sort_index(axis=0).sort_index(axis=1)

# Step 4: Plot the cross-correlation matrix with equal ordering on X and Y axes
sns.clustermap(
    correlation_matrix,
    annot=False,  # Show the correlation values on the heatmap
    cmap='coolwarm',  # Color map for the heatmap
    figsize=(40, 40),  # Adjust the figure size
    method='average',  # Hierarchical clustering method (average linkage)
    metric='correlation',  # Distance metric for clustering
    row_cluster=True,  # Enable clustering on rows
    col_cluster=True,  # Disable clustering on columns (set to True for clustering on columns)
    cbar_kws={'label': 'Correlation'},  # Label for the color bar
    vmin=0.9,
    vmax=1,
)

# Save the plot with dendrogram and cross-correlation matrix
cross_corr_output_path = os.path.join(output_dir, "cross_correlation_clusters_with_clustering.png")
plt.savefig(cross_corr_output_path)
plt.savefig(os.path.join(output_dir, "cross_correlation_clusters_with_clustering.svg"))
plt.close()

print(f"Cross-correlation plot with hierarchical clustering saved to: {cross_corr_output_path}")
