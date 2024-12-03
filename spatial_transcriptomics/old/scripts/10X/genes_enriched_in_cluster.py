import os
import json
import requests
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc  # For garbage collection to free memory

import utils.utils as ut

# Define directories and dataset information
saving_dir = ut.create_dir("/mnt/data/Thomas/PPN_CUN")

download_base = r'/mnt/data/Thomas/data'  # Path to the data on the local drive

# Manifest URL and metadata loading
url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
manifest = json.loads(requests.get(url).text)  # Load the manifest

# Fetch metadata structure from the manifest for 10Xv3
dataset_id = "WMB-10X"  # Define the dataset
metadata_json = manifest['file_listing'][dataset_id]['metadata']  # Metadata for cells
dataset_id = "WMB-10Xv3"  # Define the dataset
metadata_exp = manifest['file_listing'][dataset_id]['expression_matrices']

# Path to metadata file (cell metadata)
metadata_relative_path = metadata_json['cell_metadata_with_cluster_annotation']['files']['csv']['relative_path']
metadata_file = os.path.join(download_base, metadata_relative_path)  # Path to metadata file
exp = pd.read_csv(metadata_file)  # Load cell metadata
exp.set_index('cell_label', inplace=True)

# Define clusters of interest
sorted_and_enriched_clusters = pd.read_csv("/mnt/data/Thomas/PPN_CUN/results/3d_views/cluster_labels_and_percentages.csv")
defined_cluster_names = [x for x, y in zip(sorted_and_enriched_clusters["Label"],
                                           sorted_and_enriched_clusters["Percentage"]) if y > 50]
defined_cluster_names = ["3331 CUN-PPN Evx2 Meis2 Glut_1"]
# Sampling rate for out-cluster cells to avoid memory overload
sampling_rate = 50  # Sampling rate for out-cluster cells

# Load gene metadata to map gene identifiers to gene symbols
metadata_genes_relative_path = metadata_json['gene']['files']['csv']['relative_path']
metadata_gene_file = os.path.join(download_base, metadata_genes_relative_path)  # Path to metadata file
genes_metadata = pd.read_csv(metadata_gene_file)  # Load gene metadata
genes_metadata = genes_metadata.rename(columns={'gene_identifier': 'gene_id', 'gene_symbol': 'gene_acronym'})  # Rename for clarity

# Loop over each defined cluster
for cluster_name in defined_cluster_names:
    print(f"Processing cluster: {cluster_name}")

    # Mask for in-cluster cells
    cluster_mask = exp["cluster"] == cluster_name
    library_mask = exp["library_method"] == "10Xv3"
    cluster_and_library_mask = np.logical_and(cluster_mask, library_mask)

    # Fetch in-cluster and out-cluster cell labels
    cell_labels_in = exp[cluster_and_library_mask].index
    cell_labels_out = exp[~cluster_and_library_mask][::sampling_rate].index  # Sample out-cluster cells

    adata_cluster_in = []  # Gene expression for in-cluster cells
    adata_cluster_out = []  # Gene expression for out-cluster cells

    # Load the gene expression data for each sub-dataset and filter by in/out cluster cells
    for dsn in manifest['file_listing'][dataset_id]['expression_matrices']:
        print(f"Loading 10x data for: {dsn}")
        try:
            adata = anndata.read_h5ad(
                os.path.join(download_base, manifest['file_listing'][dataset_id]['expression_matrices'][dsn]["log2"]["files"]["h5ad"]["relative_path"]), backed='r'
            )
        except Exception as e:
            print(f"Error loading dataset {dsn}: {e}")
            continue  # Skip this dataset if there's an error

        # Mask for in-cluster and out-cluster cells
        mask_in = adata.obs.index.isin(cell_labels_in)
        mask_out = adata.obs.index.isin(cell_labels_out)

        adata_cluster_in.append(adata[mask_in])
        adata_cluster_out.append(adata[mask_out])

    print("Combining datasets!")
    # Combine datasets and filter out empty AnnData objects
    adata_cluster_in_filtered = [x for x in adata_cluster_in if len(x) > 0]
    adata_cluster_out_filtered = [x for x in adata_cluster_out if len(x) > 0]

    if not adata_cluster_in_filtered or not adata_cluster_out_filtered:
        print(f"No data to process for cluster {cluster_name}. Skipping...")
        continue

    combined_adata_in = anndata.concat(adata_cluster_in_filtered, axis=0)
    combined_adata_out = anndata.concat(adata_cluster_out_filtered, axis=0)

    # Concatenate the in and out datasets for comparison
    combined_data = combined_adata_in.concatenate(combined_adata_out, batch_key='dataset',
                                                  batch_categories=['in', 'out'])

    # Normalize and log-transform the concatenated data
    print(f"Normalizing dataset for cluster: {cluster_name}")
    sc.pp.normalize_total(combined_data, target_sum=1e4)
    sc.pp.log1p(combined_data)

    # Perform differential expression analysis (in-cluster vs. out-cluster)
    print(f"Ranking genes for cluster: {cluster_name}")
    sc.tl.rank_genes_groups(combined_data, groupby='dataset', groups=['in'], reference='out')

    # Extract all the ranked genes
    ranked_genes = pd.DataFrame({
        'gene': combined_data.uns['rank_genes_groups']['names']['in'],
        'logfoldchange': combined_data.uns['rank_genes_groups']['logfoldchanges']['in'],
        'p_value': combined_data.uns['rank_genes_groups']['pvals']['in']
    })

    # Filter out non-significant genes with p-value > 0.05
    ranked_genes_filtered = ranked_genes[ranked_genes['p_value'] <= 0.05]

    # Define a small scalar (epsilon) to avoid log(0)
    epsilon = 1e-300  # This value should be small enough to not significantly alter the results but avoid infinity

    # Add combined score (logfoldchange * -log10(p-value))
    ranked_genes_filtered['combined_score'] = ranked_genes_filtered['logfoldchange'] * -np.log10(ranked_genes_filtered['p_value'] + epsilon)

    # Merge gene metadata to get gene acronyms
    ranked_genes_filtered = ranked_genes_filtered.merge(genes_metadata[['gene_id', 'gene_acronym']], left_on='gene', right_on='gene_id', how='left')

    # Sort by combined score
    ranked_genes_filtered = ranked_genes_filtered.sort_values(by='combined_score', ascending=False)

    # Save filtered genes to a CSV file
    cluster_csv_path = os.path.join(saving_dir, f"differentially_expressed_genes_{cluster_name}.csv")
    ranked_genes_filtered.to_csv(cluster_csv_path, index=False)
    print(f"Filtered results (p <= 0.05) saved for cluster {cluster_name} in: {cluster_csv_path}")

    # Extract the top 20 ranked genes for the bar plot
    top_20_genes = ranked_genes_filtered.head(20)

    # Plot a vertical monochromatic barplot of the top 20 differentially expressed genes by combined score
    print(f"Generating barplot for top 20 genes in cluster: {cluster_name}")
    plt.figure(figsize=(20, 6))
    sns.barplot(x='gene_acronym', y='combined_score', data=top_20_genes, color='gray', edgecolor="black",
                linewidth=0.1, order=top_20_genes['gene_acronym'])  # Monochromatic bar plot
    plt.title(f'Top 20 Differentially Expressed Genes in {cluster_name}')
    plt.xlabel('Gene Acronym')
    plt.ylabel('Combined Score (Fold Change * -log10(p-value))')

    # Rotate x-tick labels for better readability
    plt.xticks(rotation=90)

    # Save the barplot
    output_dir = os.path.join(saving_dir, "scanpy_barplots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, f"{cluster_name}_diff_expr_barplot_top20.png"))
    plt.savefig(os.path.join(output_dir, f"{cluster_name}_diff_expr_barplot_top20.svg"))
    plt.close()

    # Clean up memory after processing each cluster
    del combined_adata_in, combined_adata_out, combined_data
    gc.collect()

print("Processing complete!")
