import os

import json

import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import anndata
import scanpy as sc
from concurrent.futures import ProcessPoolExecutor

from spatial_transcriptomics.old.to_check.analysis import sc_helper_functions as sc_utils

matplotlib.use("Agg")
# matplotlib.use("Qt5Agg")

########################################################################################################################
# Selected cluster and category to analyze
########################################################################################################################

genes_to_plot = "Drd1"
saving_dir = fr"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results\10X_mapped_gene_expression\{genes_to_plot}"  # Saving directory

# Name of the cluster and category: class/subclass/supertype/cluster
cluster_names = [
                 # "0951 STR D1 Gaba_3",
                 # "0962 STR D1 Gaba_8",
                 # "0961 STR D1 Gaba_8",
                 "0960 STR D1 Gaba_7",
                 ]

categories = [
              # "cluster",
              # "cluster",
              "cluster",
              ]

for cluster_name, category in zip(cluster_names, categories):

    ########################################################################################################################
    # Highlight a specific cell class/subclass/supertype/cluster in the 10x dataset
    ########################################################################################################################

    dataset_id = "WMB-10X"  # Select the dataset
    download_base = r'E:\tto\spatial_transcriptomics'  # Path to data on the local drive
    saving_folder = os.path.join(saving_dir, f"{category}_{cluster_name}")
    if not os.path.exists(saving_folder):
        os.mkdir(saving_folder)
    url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'  # Manifest url
    manifest = json.loads(requests.get(url).text)  # Load the manifest
    metadata_json = manifest['file_listing'][dataset_id]['metadata']  # Fetch metadata structure

    metadata_relative_path = metadata_json['cell_metadata_with_cluster_annotation']['files']['csv']['relative_path']
    metadata_file = os.path.join(download_base, metadata_relative_path)  # Path to metadata file
    exp = pd.read_csv(metadata_file)  # Load metadata
    exp.set_index('cell_label', inplace=True)  # Set cell_label as dataframe index

    cluster_mask = exp[category] == cluster_name  # Mask of the cells belonging to the cluster

    # Fixme: This has to be fixed as only the 10Xv3 dataset is fetched (the largest). 10Xv2 and 10XMulti or omitted
    library_mask = exp["library_method"] == "10Xv3"  # Mask of the library data

    # Combined mask of cells in the dataset that belong to the selected cluster
    cluster_and_library_mask = np.logical_and(cluster_mask, library_mask)
    # Combined mask of cells in the dataset that do not belong to the selected cluster
    not_cluster_and_library_mask = np.logical_and(~cluster_mask, library_mask)

    color = exp[f"{category}_color"][cluster_and_library_mask][0]  # Fetch the hex color of the selected cluster

    # Plotting all the cells and highlight the selected cluster (UMAP embedding)
    fig, ax = plt.subplots()  # Create figure
    fig.set_size_inches(8, 8)  # Set figure size (inches)

    # plt.scatter(exp['x'][not_cluster_and_library_mask][::10],
    #             exp['y'][not_cluster_and_library_mask][::10],
    #             s=0.1, c="#E7E7E7", alpha=1, marker=".")  # Plot the cells out of the cluster

    # plt.scatter(exp['x'],
    #             exp['y'],
    #             s=0.1, c="#E7E7E7", alpha=1, edgecolors="none")  # Plot the cells out of the cluster

    plt.scatter(exp['x'][cluster_and_library_mask],
                exp['y'][cluster_and_library_mask],
                s=0.1, c=color, edgecolors="none")  # Plot the cells in the cluster

    ax.axis('equal')  # Set aspect ratio
    ax.set_xlim(-18, 27)  # Set x axis limits
    ax.set_ylim(-18, 27)  # Set y axis limits
    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks

    res = ax.set_title(f"{cluster_name}_{category}")  # Set title
    plt.savefig(os.path.join(saving_folder, f"10x_highlight_{cluster_name}_{category}.png"), dpi=300)  # Save result
    plt.savefig(os.path.join(saving_folder, f"10x_highlight_{cluster_name}_{category}.svg"), dpi=300)  # Save result
    # plt.show()  # Show plot

    ########################################################################################################################
    # Get top X expressed genes in  the selected cluster
    ########################################################################################################################

    # Fixme: This should be fixed as only the 10Xv3 dataset is fetched (the largest). 10Xv2 and 10XMulti or omitted
    dataset_id = "WMB-10Xv3"  # Dataset name
    metadata_exp = manifest['file_listing'][dataset_id]['expression_matrices']

    # Fetch gene expression data from the selected cluster
    sampling_rate = 50  # Sampling rate to fetch sparse data (too large otherwise). Don't lower this number too much
    cell_labels_in = exp[cluster_and_library_mask].index  # Cell labels in the selected cluster
    cell_labels_out = exp[not_cluster_and_library_mask][
                      ::sampling_rate].index  # Cell labels out of the selected cluster
    adata_cluster_in = []  # Gene expression of cells in the selected cluster for each sub-dataset
    adata_cluster_out = []  # Gene expression of cells out of the selected cluster for each sub-dataset

    # Loop over each sub-dataset (.h5ad files)
    for dsn in metadata_exp:
        print(f"Loading 10x data for: {dsn}")
        adata = anndata.read_h5ad(os.path.join(download_base,
                                               metadata_exp[dsn]["log2"]["files"]["h5ad"]["relative_path"]),
                                  backed='r')
        mask_in = adata.obs.index.isin(cell_labels_in)  # Creates mask for the cells in the selected cluster
        mask_out = adata.obs.index.isin(cell_labels_out)  # Creates mask for the cells out of the selected cluster
        adata_cluster_in.append(adata[mask_in])  # Appends result
        adata_cluster_out.append(adata[mask_out])  # Appends result

    print("Combining all 10x datasets!")
    adata_cluster_in_filtered = [x for x in adata_cluster_in if len(x) > 0]  # Filters out empty anndata
    adata_cluster_out_filtered = [x for x in adata_cluster_out if len(x) > 0]  # Filters out empty anndata
    combined_adata_in = anndata.concat(adata_cluster_in_filtered, axis=0)  # Combine all the datasets
    combined_adata_out = anndata.concat(adata_cluster_out_filtered, axis=0)  # Combine all the datasets

    combined_data_in = combined_adata_in.to_df()  # Load the expression data into memory
    combined_data_out = combined_adata_out.to_df()  # Load the expression data into memory

    print("Computing enriched genes")
    combined_data = combined_adata_in.concatenate(combined_adata_out,
                                                  batch_key='dataset',
                                                  batch_categories=['in', 'out'])  # Concatenate the datasets
    sc.pp.normalize_total(combined_data, target_sum=1e4)  # Normalize the datasets
    sc.pp.log1p(combined_data)  # Compute the log1p of the datasets
    # sc.pp.highly_variable_genes(combined_data, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.tl.rank_genes_groups(combined_data, groupby='dataset', groups=['in'],
                            reference='out')  # Rank genes by enrichment






    # Filter for genes enriched in 'in' group
    log_fold_changes = pd.DataFrame(combined_data.uns['rank_genes_groups']['logfoldchanges'])['in']
    pvals_adj = pd.DataFrame(combined_data.uns['rank_genes_groups']['pvals_adj'])['in']

    # Filtering for enriched genes
    number_of_enriched_genes = 25  # Number of top genes to plot
    gene_names = adata.var['gene_symbol']  # All gene names
    sorted_gene_names = [gene_names[gene_names.index == g[0]].values[0] for g in
                         combined_data.uns['rank_genes_groups']["names"]]

    df = pd.DataFrame({
        'gene_name': sorted_gene_names,
        'log_fold_change': log_fold_changes,
        'adjusted_p_value': pvals_adj,
        # 'highly_variable': np.array(combined_data.var['highly_variable']),
    })

    # Fixme: Potentially problematic as the value is subjective
    expression_threshold = 3  # You can adjust this threshold
    # adata_in_normalized = sc_utils.min_max_normalize(combined_data[combined_data.obs["dataset"] == "in"])
    # adata_in_normalized = sc_utils.min_max_normalize(combined_adata_in)
    expressed_cells = (combined_adata_in.X.toarray() > expression_threshold).mean(axis=0)  # For a sparse matrix, use .A1 after .mean()
    # expressed_cells = (adata_in_normalized.X > expression_threshold).mean(axis=0)  # For a sparse matrix, use .A1 after .mean()
    df['percent_cells_expressed'] = expressed_cells
    # df['percent_cells_expressed'] = np.array(expressed_cells)[0]

    # Filter for genes that are expressed in at least a certain percentage of cells
    percent_cells_threshold = 0.5  # Adjust this threshold as needed
    n_valid_genes = df['percent_cells_expressed'] > percent_cells_threshold
    print(f"{np.sum(n_valid_genes)} genes meeting criteria: More than {percent_cells_threshold*100}% cells exp>{expression_threshold}")
    enriched_genes_consistent = df[(df['log_fold_change'] > 0) &
                                   (df['adjusted_p_value'] < 0.05) &
                                    n_valid_genes]
    # enriched_genes_consistent = df[(df['log_fold_change'] > 0) &
    #                                (df['adjusted_p_value'] < 0.05)]

    # Sort by log fold change
    enriched_genes_df_sorted = enriched_genes_consistent.sort_values(by='log_fold_change', ascending=False)
    # enriched_genes_df_sorted = enriched_genes_consistent.sort_values(by='adjusted_p_value', ascending=False)

    #enriched_genes_df = df[(df['log_fold_change'] > 0) & (df['adjusted_p_value'] < 0.05)]
    #enriched_genes_df_sorted = enriched_genes_df.sort_values(by='log_fold_change', ascending=False)
    enriched_gene_names_sorted = enriched_genes_df_sorted['gene_name'].tolist()

    # Plotting
    plt.figure(figsize=(15, 6))  # Create figure
    ax = plt.subplot(111)  # Create subplot
    ax.bar(np.arange(0, len(enriched_genes_df_sorted["log_fold_change"][:number_of_enriched_genes]), 1),
           enriched_genes_df_sorted["log_fold_change"][:number_of_enriched_genes])  # Create the bar plot
    ax.set_xticks(
        np.arange(0, len(enriched_genes_df_sorted["log_fold_change"][:number_of_enriched_genes]), 1))  # Set x ticks
    ax.set_xticklabels(enriched_gene_names_sorted[:number_of_enriched_genes])  # Set x tick labels
    plt.title(f'Top enriched genes for {category} {cluster_name}')  # Set the title
    plt.xlabel('Genes')  # Set x label
    plt.xticks(rotation=45)
    plt.ylabel('Log fold-change expression')  # Set y label
    plt.tight_layout()
    plt.savefig(os.path.join(saving_folder,
                             f"10x_top_{number_of_enriched_genes}_enriched_genes_{cluster_name}_{category}.png"),
                dpi=300)
    plt.savefig(os.path.join(saving_folder,
                             f"10x_top_{number_of_enriched_genes}_enriched_genes_{cluster_name}_{category}.svg"),
                dpi=300)  # Save result

    ########################################################################################################################
    # Highlight gene expression levels in the 10x dataset
    ########################################################################################################################

    # Pre-loading dataset paths
    datasets_paths = {dsn: os.path.join(download_base, metadata_exp[dsn]["log2"]["files"]["h5ad"]["relative_path"]) for
                      dsn in metadata_exp}

    # Using ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=None) as executor:
        for n, gene_name in enumerate(enriched_gene_names_sorted[:number_of_enriched_genes]):
            executor.submit(sc_utils.process_gene, gene_name, datasets_paths, exp, saving_folder, cluster_name,
                            category, n)
