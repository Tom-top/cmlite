"""
This script

Author: Thomas Topilko
"""


import os
import json
import requests
import numpy as np
import pandas as pd
import tifffile
import anndata
import matplotlib

matplotlib.use("Agg")

import utils.utils as ut

from spatial_transcriptomics.old.utils.coordinate_manipulation import filter_points_in_3d_mask

atlas_used = "gubra"
datasets = np.arange(1, 6, 1)
n_datasets = len(datasets)

annotation_directory = fr"resources{os.sep}atlas"
annotation_path = os.path.join(annotation_directory, f"{atlas_used}_annotation_mouse.tif")
annotation = np.transpose(tifffile.imread(annotation_path), (1, 2, 0))
annotation_json = os.path.join(annotation_directory, f"{atlas_used}_annotation_mouse.json")

download_base = r"/default/path"  # PERSONAL
map_directory = ut.create_dir(rf"/default/path")  # PERSONAL

brain_mask = tifffile.imread(os.path.join(map_directory, r"hemisphere_mask.tif"))
results_directory = os.path.join(map_directory, "results")
saving_directory = ut.create_dir(os.path.join(results_directory, "3d_views"))
transform_directory = r"resources/abc_atlas"

url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
manifest = json.loads(requests.get(url).text)
data_id_10X = "WMB-10X"  # Select the dataset
metadata_json = manifest['file_listing'][data_id_10X]['metadata']  # Fetch metadata structure
metadata_relative_path = metadata_json['cell_metadata_with_cluster_annotation']['files']['csv']['relative_path']
metadata_file = os.path.join(download_base, metadata_relative_path)  # Path to metadata file
exp = pd.read_csv(metadata_file, low_memory=False)  # Load metadata
exp.set_index('cell_label', inplace=True)  # Set cell_label as dataframe index

metadata_genes_relative_path = metadata_json['gene']['files']['csv']['relative_path']
metadata_gene_file = os.path.join(download_base, metadata_genes_relative_path)  # Path to metadata file
genes = pd.read_csv(metadata_gene_file)  # Load metadata

########################################################################################################################
# LOAD THE SINGLE-CELL DATASETS FOR EACH CHEMISTRY
########################################################################################################################

single_cell_chemistry = ["WMB-10Xv3", "WMB-10Xv2", "WMB-10XMulti"]
single_cell_datasets = {}

for modality in single_cell_chemistry:
    single_cell_datasets[modality] = []
    metadata_exp = manifest['file_listing'][modality]['expression_matrices']
    print("")
    for dsn in metadata_exp:
        ut.print_c(f"[INFO] Loading 10x data for: {dsn}", end="\r")
        adata = anndata.read_h5ad(os.path.join(download_base,
                                               metadata_exp[dsn]["log2"]["files"]["h5ad"]["relative_path"]),
                                  backed='r')
        single_cell_datasets[modality].append(adata)


####################################################################################################################
# COMPUTE THE UNIQUE CELL CLUSTERS
####################################################################################################################

# Create buffers for the merged datasets
# Cluster
cells_cluster_merged = []

print("")
for i, dataset_n in enumerate(datasets):

    ut.print_c(f"[INFO] Loading MERFISH dataset: {dataset_n}")

    # Select the correct dataset
    if dataset_n < 5:
        dataset_id = f"Zhuang-ABCA-{dataset_n}"
    else:
        dataset_id = f"MERFISH-C57BL6J-638850"
    metadata = manifest['file_listing'][dataset_id]['metadata']

    # Fetch labels for each cell in the selected dataset
    metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']
    cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
    cell_metadata_file_ccf = os.path.join(download_base, cell_metadata_path_ccf)
    cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
    cell_metadata_ccf.set_index('cell_label', inplace=True)
    cell_labels = cell_metadata_ccf.index

    # Fetch metadata ofr each for each cell in the selected dataset (class, subclass, supertype, cluster...)
    metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
    cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
    cell_metadata_file_views = os.path.join(download_base, cell_metadata_path_views)
    cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
    cell_metadata_views.set_index('cell_label', inplace=True)

    # Fetch the transformed coordinates from the selected dataset
    transformed_coordinates = np.load(
        os.path.join(transform_directory, f"all_transformed_cells_{atlas_used}_{dataset_n}.npy"))

    # Pre-calculate the chunks to run through in the selected dataset
    chunk_size = 10  # Size of each data chunk (voxels)
    chunks_start = np.arange(0, brain_mask.shape[0], chunk_size)
    chunks_end = np.arange(chunk_size, brain_mask.shape[0], chunk_size)
    if chunks_end[-1] != brain_mask.shape[0]:
        chunks_end = np.append(chunks_end, brain_mask.shape[0])
    n_chunks = len(chunks_start)  # Number of chunks

    # Create buffers for the selected dataset
    # Cluster
    cells_cluster_dataset = []

    ################################################################################################################
    # ITERATE OVER EVERY CHUNK IN THE SELECTED DATASET
    ################################################################################################################

    for n, (cs, ce) in enumerate(zip(chunks_start, chunks_end)):
        ut.print_c(f"[INFO] Processing chunk: {cs}:{ce}. {n}/{n_chunks}", end="\r")

        # Generate chunk mask
        chunk_mask = brain_mask.copy()
        chunk_mask[0:cs] = 0
        chunk_mask[ce:] = 0

        # Fetch the cell coordinates within the chunk
        _, mask_point = filter_points_in_3d_mask(transformed_coordinates, chunk_mask)
        filtered_labels = np.array(cell_labels)[::-1][mask_point]
        filtered_metadata_views = cell_metadata_views.loc[filtered_labels]

        # Extract data for each category
        # Cluster
        cells_cluster = filtered_metadata_views["cluster"].tolist()
        cells_cluster_dataset.extend(cells_cluster)

    # Cluster
    cells_cluster_merged.extend(cells_cluster_dataset)

# Get unique occurrences in each category
unique_cells_cluster = np.unique(cells_cluster_merged, return_index=True)  # Cluster

########################################################################################################################
# BLABLA
########################################################################################################################

target_genes = genes["gene_symbol"]
n_target_genes = len(target_genes)

df_to_save = {}
duplicate_gene_entries = {}

unique_clusters = unique_cells_cluster[0][5000:]
n_unique_cluster = len(unique_clusters)

for n, cluster_name in enumerate(unique_clusters):

    ut.print_c(f"[INFO] Fetching data for cluster: {cluster_name}; {n + 1}/{n_unique_cluster}")
    cluster_mask = exp["cluster"] == cluster_name  # Mask of the cells belonging to the cluster

    # Iterate over all library methods (10Xv3, 10Xv2, 10xRSeq_Mult)
    df_to_save[cluster_name] = {}

    # Combined mask of cells in the dataset that belong to the selected cluster
    combined_data_in_all = []

    for library_method in single_cell_chemistry:

        if library_method == "WMB-10Xv3":
            chemistry_id = "10Xv3"
        elif library_method == "WMB-10Xv2":
            chemistry_id = "10Xv2"
        elif library_method == "WMB-10XMulti":
            chemistry_id = "10xRSeq_Mult"

        library_mask = exp["library_method"] == chemistry_id  # Mask of the library data
        cluster_and_library_mask = np.logical_and(cluster_mask, library_mask)

        if np.sum(cluster_and_library_mask) > 0:
            # Fetch cell labels in the selected cluster and library
            cell_labels_in = exp[cluster_and_library_mask].index

            # Collect gene expression data for the current library
            adata_cluster_in = []
            for adata in single_cell_datasets[library_method]:
                mask_in = adata.obs.index.isin(cell_labels_in)
                adata_cluster_in.append(adata[mask_in])

            # Filter out empty datasets
            adata_cluster_in_filtered = [x for x in adata_cluster_in if len(x) > 0]
            if adata_cluster_in_filtered:
                combined_adata_in = anndata.concat(adata_cluster_in_filtered, axis=0)
                combined_data_in_all.append(combined_adata_in.to_df())  # Store for later processing

    # Combine data from all libraries
    if combined_data_in_all:
        combined_data_in = pd.concat(combined_data_in_all, axis=0)  # Merge all library data

        # Calculate mean expression for each target gene
        for m, target_gene in enumerate(target_genes):

            ut.print_c(
                f"[INFO {cluster_name}] Fetching expression data for gene: {target_gene}; {m + 1}/{len(target_genes)}",
                end="\r")
            gene_mask = genes["gene_symbol"] == target_gene
            gene_ids = genes["gene_identifier"][gene_mask]
            n_gene_entries = len(gene_ids)

            for gene_id in gene_ids:
                # Calculate the mean gene expression
                mean_gene_expression = float(np.mean(combined_data_in[gene_id], axis=0))
                df_to_save[cluster_name][gene_id] = mean_gene_expression
    else:
        ut.print_c(f"[WARNING {cluster_name}] No overlap detected between the cluster and any library data!")
        for target_gene in target_genes:
            gene_mask = genes["gene_symbol"] == target_gene
            gene_ids = genes["gene_identifier"][gene_mask]
            n_gene_entries = len(gene_ids)
            for gene_id in gene_ids:
                df_to_save[cluster_name][gene_id] = np.nan

    if (n + 1) % 500 == 0 or n + 1 == n_unique_cluster:
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame.from_dict(df_to_save, orient='index')
        df_dup = pd.DataFrame.from_dict(duplicate_gene_entries, orient='index')
        # Assign a name to the index
        df.index.name = 'cluster_name'
        df_dup.index.name = 'gene_name'
        # Save the DataFrame to a CSV file
        df.to_csv(os.path.join(saving_directory, f"cluster_log2_mean_gene_expression_{n + 1 - 500}-{n + 1}.csv"))
        df_dup.to_csv(os.path.join(saving_directory, f"duplicate_gene_entries_{n + 1 - 500}-{n + 1}.csv"))
        df_to_save = {}
        duplicate_gene_entries = {}
