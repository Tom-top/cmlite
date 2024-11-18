"""
This script generates visualizations showing the distribution of the cells within the combined datasets
 in a restricted binary mask.
For each dataset, coronal, sagittal, and horizontal views are produced with the Gubra LSFM reference in the background.
The cells are labeled according to the ABC atlas ontology (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas).
The source of the data is from Zhang M. et al., in Nature, 2023 (DOI: 10.1038/s41586-023-06808-9).

Author: Thomas Topilko
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import tifffile
from collections import Counter
import anndata
# from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

import utils.utils as ut

from spatial_transcriptomics.old.utils.coordinate_manipulation import filter_points_in_3d_mask
import spatial_transcriptomics.old.utils.plotting as st_plt

ATLAS_USED = "gubra"
DATASETS = np.arange(1, 6, 1)
N_DATASETS = len(DATASETS)
CATEGORY_NAMES = ["neurotransmitter", "class", "subclass", "supertype", "cluster"]
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]
BILATERAL = True  # If True: generate bilateral cell distribution in the 3D representations
ONLY_NEURONS = False  # If True: only generate plots for neurons, excluding all non-neuronal cells
PLOT_MOST_REPRESENTED_CATEGORIES = False
PERCENTAGE_THRESHOLD = 50
categories = ["cluster"]

ANO_DIRECTORY = r"resources\atlas"
ANO_PATH = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.tif")
ANO = np.transpose(tifffile.imread(ANO_PATH), (1, 2, 0))
ANO_JSON = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.json")

DOWNLOAD_BASE = r"E:\tto\spatial_transcriptomics"  # PERSONAL
MAP_DIR = ut.create_dir(rf"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression")  # PERSONAL
WHOLE_REGION = True  # If true, the unprocessed mask will be used
LABELED_MASK = False  # If true the TISSUE_MASK is a labeled 32bit mask, not a binary.
PLOT_COUNTS_BY_CATEGORY = True  # If true plots the category plot
SHOW_OUTLINE = False
ZOOM = False
SHOW_REF = False

if WHOLE_REGION:
    LABELED_MASK = False
if LABELED_MASK:  # Each label will be processed separately.
    TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"labeled_mask.tif"))
    unique_labels = np.unique(TISSUE_MASK)
    TISSUE_MASKS = [(TISSUE_MASK == ul).astype("uint8") * 255 for ul in unique_labels if not ul == 0]
    labels = [ul for ul in unique_labels if not ul == 0]
else:
    labels = [1]
    if WHOLE_REGION:
        TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"hemisphere_mask.tif"))
        # TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"smoothed_mask.tif"))
    else:
        TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"smoothed_mask.tif"))
    TISSUE_MASKS = [TISSUE_MASK]

MAIN_RESULTS_DIR = os.path.join(MAP_DIR, "results")
RESULTS_DIR = ut.create_dir(os.path.join(MAIN_RESULTS_DIR, "3d_views"))

TRANSFORM_DIR = r"resources/abc_atlas"
REFERENCE_FILE = fr"resources/atlas/{ATLAS_USED}_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)
REFERENCE_SHAPE = REFERENCE.shape

ABC_ATLAS_DIRECTORY = r"resources\abc_atlas"
CLASS_COUNTS = pd.read_excel(os.path.join(ABC_ATLAS_DIRECTORY, "counts_cells_class.xlsx"))
SUBCLASS_COUNTS = pd.read_excel(os.path.join(ABC_ATLAS_DIRECTORY, "counts_cells_subclass.xlsx"))
SUPERTYPE_COUNTS = pd.read_excel(os.path.join(ABC_ATLAS_DIRECTORY, "counts_cells_supertype.xlsx"))
CLUSTER_COUNTS = pd.read_excel(os.path.join(ABC_ATLAS_DIRECTORY, "counts_cells_cluster.xlsx"))
NEUROTRANSMITTER_COUNTS = pd.read_excel(os.path.join(ABC_ATLAS_DIRECTORY, "counts_cells_neurotransmitter.xlsx"))

url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
manifest = json.loads(requests.get(url).text)
data_id_10X = "WMB-10X"  # Select the dataset
metadata_json = manifest['file_listing'][data_id_10X]['metadata']  # Fetch metadata structure
metadata_relative_path = metadata_json['cell_metadata_with_cluster_annotation']['files']['csv']['relative_path']
metadata_file = os.path.join(DOWNLOAD_BASE, metadata_relative_path)  # Path to metadata file
exp = pd.read_csv(metadata_file, low_memory=False)  # Load metadata
exp.set_index('cell_label', inplace=True)  # Set cell_label as dataframe index

metadata_genes_relative_path = metadata_json['gene']['files']['csv']['relative_path']
metadata_gene_file = os.path.join(DOWNLOAD_BASE, metadata_genes_relative_path)  # Path to metadata file
genes = pd.read_csv(metadata_gene_file)  # Load metadata

# Fixme: This should be fixed as only the 10Xv3 dataset is fetched (the largest). 10Xv2 and 10XMulti or omitted
dataset_id = "WMB-10Xv3"  # Dataset name
metadata_exp = manifest['file_listing'][dataset_id]['expression_matrices']
adatas = []
print("")
for dsn in metadata_exp:
    ut.print_c(f"[INFO] Loading 10x data for: {dsn}", end="\r")
    adata = anndata.read_h5ad(os.path.join(DOWNLOAD_BASE,
                                           metadata_exp[dsn]["log2"]["files"]["h5ad"]["relative_path"]),
                              backed='r')
    adatas.append(adata)
colormap = plt.get_cmap("YlOrBr")

# mean_gene_expression_data = pd.read_feather(r"resources\abc_atlas\cluster_log2_mean_gene_expression_merge.feather")

########################################################################################################################
# ITERATE OVER EVERY BLOB (LABEL)
########################################################################################################################

for ul, TISSUE_MASK in zip(labels, TISSUE_MASKS):

    region_id_counts = Counter(ANO[TISSUE_MASK == 255])

    most_common_region_id = max(region_id_counts, key=region_id_counts.get)
    structure = ut.read_ano_json(ANO_JSON)
    region_acronym = ut.find_key_by_id(structure, most_common_region_id, key="acronym")

    with open(ANO_JSON, 'r') as f:
        json_data = json.load(f)
    structure = json_data['msg'][0]

    if LABELED_MASK:
        SAVING_DIR = os.path.join(RESULTS_DIR, f"{region_acronym}_blob_1")
        if os.path.exists(SAVING_DIR):
            n_region_dirs = len([n for n in os.listdir(RESULTS_DIR) if n.startswith(region_acronym)])
            SAVING_DIR = os.path.join(RESULTS_DIR, f"{region_acronym}_blob_{n_region_dirs + 1}")
        SAVING_DIR = ut.create_dir(SAVING_DIR)
    else:
        SAVING_DIR = RESULTS_DIR

    sorted_region_id_counts = dict(sorted(region_id_counts.items(), key=lambda item: item[1], reverse=True))
    ids = list(sorted_region_id_counts.keys())

    # Divide the values by 2 as we are working on hemispheres
    region_id_counts_total = [int(np.sum(ANO == id) / 2) for id in ids]
    acros = [ut.find_key_by_id(structure, id, key="acronym") for id in ids]
    colors = [ut.hex_to_rgb(ut.find_key_by_id(structure, id, key="color_hex_triplet")) for id in ids]
    st_plt.stacked_bar_plot_atlas_regions(sorted_region_id_counts,
                                          np.array(region_id_counts_total),
                                          np.array(acros),
                                          np.array(colors),
                                          SAVING_DIR)

    # Create buffers for the merged datasets
    filtered_points_merged = []
    # Class
    cells_class_merged = []
    # Cluster
    cells_cluster_merged = []

    ####################################################################################################################
    # ITERATE OVER EVERY DATASET
    ####################################################################################################################

    print("")
    for i, dataset_n in enumerate(DATASETS):

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
        cell_metadata_file_ccf = os.path.join(DOWNLOAD_BASE, cell_metadata_path_ccf)
        cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
        cell_metadata_ccf.set_index('cell_label', inplace=True)
        cell_labels = cell_metadata_ccf.index

        # Fetch metadata ofr each for each cell in the selected dataset (class, subclass, supertype, cluster...)
        metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
        cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
        cell_metadata_file_views = os.path.join(DOWNLOAD_BASE, cell_metadata_path_views)
        cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
        cell_metadata_views.set_index('cell_label', inplace=True)

        # Fetch the transformed coordinates from the selected dataset
        # transformed_coordinates = np.load(os.path.join(TRANSFORM_DIR, f"all_transformed_cells_{ATLAS_USED}_{dataset_n}.npy"))
        transformed_coordinates = np.load(
            os.path.join(TRANSFORM_DIR, f"all_transformed_cells_{ATLAS_USED}_{dataset_n}.npy"))

        # Pre-calculate the chunks to run through in the selected dataset
        chunk_size = 10  # Size of each data chunk (voxels)
        chunks_start = np.arange(0, TISSUE_MASK.shape[0], chunk_size)
        chunks_end = np.arange(chunk_size, TISSUE_MASK.shape[0], chunk_size)
        if chunks_end[-1] != TISSUE_MASK.shape[0]:
            chunks_end = np.append(chunks_end, TISSUE_MASK.shape[0])
        n_chunks = len(chunks_start)  # Number of chunks

        # Create buffers for the selected dataset
        filtered_points_dataset = []
        # Class
        cells_class_dataset = []
        cells_class_colors_dataset = []
        # Cluster
        cells_cluster_dataset = []
        cells_cluster_colors_dataset = []

        ################################################################################################################
        # ITERATE OVER EVERY CHUNK IN THE SELECTED DATASET
        ################################################################################################################

        for n, (cs, ce) in enumerate(zip(chunks_start, chunks_end)):
            ut.print_c(f"[INFO] Processing chunk: {cs}:{ce}. {n}/{n_chunks}", end="\r")

            # Generate chunk mask
            chunk_mask = TISSUE_MASK.copy()
            chunk_mask[0:cs] = 0
            chunk_mask[ce:] = 0

            # Fetch the cell coordinates within the chunk
            filtered_points, mask_point = filter_points_in_3d_mask(transformed_coordinates, chunk_mask)
            filtered_points_dataset.extend(filtered_points)
            filtered_labels = np.array(cell_labels)[::-1][mask_point]
            filtered_metadata_views = cell_metadata_views.loc[filtered_labels]

            # Extract data for each category
            # Class
            cells_class = filtered_metadata_views["class"].tolist()
            cells_cls_color = filtered_metadata_views["class_color"].tolist()
            cells_class_dataset.extend(cells_class)
            cells_class_colors_dataset.extend(cells_cls_color)
            # Cluster
            cells_cluster = filtered_metadata_views["cluster"].tolist()
            cells_cluster_color = filtered_metadata_views["cluster_color"].tolist()
            cells_cluster_dataset.extend(cells_cluster)
            cells_cluster_colors_dataset.extend(cells_cluster_color)

        # Extend the buffers for the merged datasets
        filtered_points_merged.extend(filtered_points_dataset)
        # Class
        cells_class_merged.extend(cells_class_dataset)
        # Cluster
        cells_cluster_merged.extend(cells_cluster_dataset)

    # Get unique occurrences in each category
    # Class
    unique_cells_class = np.unique(cells_class_merged, return_index=True)
    # Cluster
    unique_cells_cluster = np.unique(cells_cluster_merged, return_index=True)

    # Create masks for neuronal cells
    non_neuronal_mask_global = np.array(
        [True if any([j in i for j in NON_NEURONAL_CELL_TYPES]) else False for i in cells_class_merged])

    ################################################################################################################
    # FETCH GENE EXPRESSION DATA FOR CLUSTER
    ################################################################################################################

    n_unique_cluster = len(unique_cells_cluster[0])
    target_genes = genes["gene_symbol"]
    n_target_genes = len(target_genes)

    df_to_save = {}
    duplicate_gene_entries = {}

    # Fixme: REMOVE THIS
    unique_clusters = unique_cells_cluster[0]

    for n, cluster_name in enumerate(unique_clusters):
        # if cluster_name not in mean_gene_expression_data["cluster_name"]:
        ut.print_c(f"[INFO] Fetching data for cluster: {cluster_name}; {n + 1}/{n_unique_cluster}")
        # cluster_name = cells_cluster_merged[cluster]
        cluster_mask = exp["cluster"] == cluster_name  # Mask of the cells belonging to the cluster

        # Fixme: This has to be fixed as only the 10Xv3 dataset is fetched (the largest). 10Xv2 and 10XMulti or omitted
        library_mask = exp["library_method"] == "10Xv3"  # Mask of the library data

        # Combined mask of cells in the dataset that belong to the selected cluster
        cluster_and_library_mask = np.logical_and(cluster_mask, library_mask)

        df_to_save[cluster_name] = {}

        if np.sum(cluster_and_library_mask) > 0:
            # Fetch gene expression data from the selected cluster
            # cell_labels_in = exp[cluster_and_library_mask].index  # Cell labels in the selected cluster
            cell_labels_in = exp[cluster_and_library_mask].index  # Cell labels in the selected cluster
            adata_cluster_in = []  # Gene expression of cells in the selected cluster for each sub-dataset

            # Loop over each sub-dataset (.h5ad files)
            for adata in adatas:
                mask_in = adata.obs.index.isin(cell_labels_in)  # Creates mask for the cells in the selected cluster
                adata_cluster_in.append(adata[mask_in])  # Appends result

            adata_cluster_in_filtered = [x for x in adata_cluster_in if len(x) > 0]  # Filters out empty anndata
            combined_adata_in = anndata.concat(adata_cluster_in_filtered, axis=0)  # Combine all the datasets

            combined_data_in = combined_adata_in.to_df()  # Load the expression data into memory

            for m, target_gene in enumerate(target_genes):
                ut.print_c(f"[INFO {cluster_name}] Fetching expression data for gene: {target_gene}; {m + 1}/{n_target_genes}", end="\r")
                gene_mask = genes["gene_symbol"] == target_gene
                gene_id = genes["gene_identifier"][gene_mask]
                n_gene_entries = len(gene_id)
                if n_gene_entries > 1:
                    ut.print_c(f"[WARNING {cluster_name}] Duplicate entry for gene: {target_gene}: {n_gene_entries}!")
                    duplicate_gene_entries[target_gene] = n_gene_entries
                    gene_id = gene_id.iloc[0]
                mean_gene_expression = float(np.mean(combined_data_in[gene_id], axis=0))
                df_to_save[cluster_name][target_gene] = mean_gene_expression
        else:
            ut.print_c(f"[WARNING {cluster_name}] No overlap detected between the cluster and the 10X data!")
            for target_gene in target_genes:
                df_to_save[cluster_name][target_gene] = np.nan

        if (n+1) % 500 == 0:
            # Convert the dictionary to a DataFrame
            df = pd.DataFrame.from_dict(df_to_save, orient='index')
            df_dup = pd.DataFrame.from_dict(duplicate_gene_entries, orient='index')
            # Assign a name to the index
            df.index.name = 'cluster_name'
            df_dup.index.name = 'gene_name'
            # Save the DataFrame to a CSV file
            df.to_csv(os.path.join(SAVING_DIR, f"cluster_log2_mean_gene_expression_{n+1-500}-{n+1}.csv"))
            df_dup.to_csv(os.path.join(SAVING_DIR, f"duplicate_gene_entries_{n+1-500}-{n+1}.csv"))
            df_to_save = {}
            duplicate_gene_entries = {}
