import os
import json
import requests
import numpy as np
import pandas as pd
import tifffile
import anndata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut

import analysis.measurements.voxelization as vox

from spatial_transcriptomics.utils.coordinate_manipulation import filter_points_in_3d_mask

ATLAS_USED = "gubra"
DATASETS = np.arange(1, 6, 1)
N_DATASETS = len(DATASETS)
CATEGORY_NAMES = ["cluster"]
NON_NEURONAL_CELL_TYPES = [["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]]
# NON_NEURONAL_CELL_TYPES = [["Astro"], ["Oligo"], ["Vascular"], ["Immune"], ["Epen"], ["OEC"]]
SHOW_GLIA = False
# target_genes = ["Hcrtr1", "Hcrtr2"]
target_genes = ["Glp1r", "Gip", "Ramp1", "Ramp2", "Ramp3", "Calca", "Hcrtr1", "Hcrtr2"]

DOWNLOAD_BASE = r"/default/path"  # PERSONAL
MAP_DIR = ut.create_dir(rf"/default/path")  # PERSONAL
TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"whole_brain_mask.tif"))
RESULTS_DIR = ut.create_dir(os.path.join(MAP_DIR, "results"))
GENE_EXPRESSION_DIR = ut.create_dir(os.path.join(RESULTS_DIR, "gene_expression"))

TRANSFORM_DIR = r"resources/abc_atlas"
REFERENCE_FILE = fr"resources/atlas/{ATLAS_USED}_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)

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
ut.print_c("[INFO] Loading mean gene expression matrix!")
mean_expression_matrix_path = r"resources\abc_atlas\cluster_log2_mean_gene_expression_merge.feather"
mean_expression_matrix = pd.read_feather(mean_expression_matrix_path)

metadata_views = []
transformed_cells = []
labels = []
for i, dataset_n in enumerate(DATASETS):

    ut.print_c(f"[INFO] Loading data from dataset: {dataset_n}")

    if dataset_n < 5:
        dataset_id = f"Zhuang-ABCA-{dataset_n}"
    else:
        dataset_id = f"MERFISH-C57BL6J-638850"
    url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
    manifest = json.loads(requests.get(url).text)
    metadata = manifest['file_listing'][dataset_id]['metadata']
    metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
    metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']
    expression_matrices = manifest['file_listing'][dataset_id]['expression_matrices']

    # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
    cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
    cell_metadata_file_ccf = os.path.join(DOWNLOAD_BASE, cell_metadata_path_ccf)
    cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
    cell_metadata_ccf.set_index('cell_label', inplace=True)
    cell_labels = cell_metadata_ccf.index
    n_cells_ccf = len(cell_metadata_ccf)
    labels.append(cell_labels)

    # Filter out the cells
    transformed_coordinates = np.load(
        os.path.join(TRANSFORM_DIR, f"all_transformed_cells_{ATLAS_USED}_{dataset_n}.npy"))
    transformed_cells.append(transformed_coordinates)

    # Views
    cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
    cell_metadata_file_views = os.path.join(DOWNLOAD_BASE, cell_metadata_path_views)
    cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
    cell_metadata_views.set_index('cell_label', inplace=True)
    metadata_views.append(cell_metadata_views)


print("")
for dsn in metadata_exp:
    ut.print_c(f"[INFO] Loading 10x data for: {dsn}")
    adata = anndata.read_h5ad(os.path.join(DOWNLOAD_BASE,
                                           metadata_exp[dsn]["log2"]["files"]["h5ad"]["relative_path"]),
                              backed='r')
    adatas.append(adata)

CHUNK_SIZE = 10
print("")

chunks_start = np.arange(0, TISSUE_MASK.shape[0], CHUNK_SIZE)
chunks_end = np.arange(CHUNK_SIZE, TISSUE_MASK.shape[0], CHUNK_SIZE)
if chunks_end[-1] != TISSUE_MASK.shape[0]:
    chunks_end = np.append(chunks_end, TISSUE_MASK.shape[0])
n_chunks = len(chunks_start)

for cell_type in NON_NEURONAL_CELL_TYPES:

    # Create buffers for the selected dataset
    filtered_points_dataset = []
    # Cluster
    cells_cluster_dataset = []
    chunk_mask = TISSUE_MASK.copy()

    for i, dataset_n in enumerate(DATASETS):

        ut.print_c(f"[INFO] Loading data from dataset: {dataset_n}")
        # Filter out the cells
        transformed_coordinates = transformed_cells[i]

        # Views
        cell_metadata_views = metadata_views[i]

        # Filter points
        filtered_points, mask_point = filter_points_in_3d_mask(transformed_coordinates, chunk_mask)
        filtered_labels = np.array(labels[i])[::-1][mask_point]
        filtered_metadata_views = cell_metadata_views.loc[filtered_labels]

        # Extract data for each category
        cell_classes = filtered_metadata_views["class"].tolist()
        cell_categories = filtered_metadata_views["cluster"].tolist()

        if SHOW_GLIA:
            non_neuronal_mask = np.array(
                [False if any([j in i for j in cell_type]) else True for i in cell_classes])
        else:
            non_neuronal_mask = np.array(
                [True if any([j in i for j in cell_type]) else False for i in cell_classes])

        if filtered_points.size > 0:
            filtered_points_dataset.append(filtered_points[~non_neuronal_mask])
            cells_cluster_dataset.append(np.array(cell_categories)[~non_neuronal_mask])
        else:
            filtered_points_dataset.append(np.array([]))
            cells_cluster_dataset.append(np.array([]))

    filtered_points_dataset = np.concatenate(filtered_points_dataset)
    cells_cluster_dataset = np.concatenate(cells_cluster_dataset)

    ############################################################################################################
    # SAVE FLYTHROUGH IN CORONAL
    ############################################################################################################

    unique_cells_cluster_colors = np.full_like(cells_cluster_dataset, 0).astype(float)

    # Filter out empty arrays
    non_empty_arrays = [arr for arr in filtered_points_dataset if arr.size > 0]
    non_empty_categories = [arr for arr in cells_cluster_dataset if len(arr) > 0]

    unique_cell_clusters = np.unique(cells_cluster_dataset, return_index=True)
    n_unique_cluster = len(unique_cell_clusters[0])

    for target_gene in target_genes:
        SAVING_DIRECTORY = ut.create_dir(GENE_EXPRESSION_DIR, target_gene)

        for n, cluster_name in enumerate(unique_cell_clusters[0]):
            cluster_mask = mean_expression_matrix["cluster_name"] == cluster_name
            try:
                mean_expression = np.array(mean_expression_matrix[cluster_mask][target_gene])[0]
                if np.isnan(mean_expression):
                    ut.print_c(f"[WARNING] Missing data for cluster {cluster_name}!")
                    mean_expression = 0
                else:
                    ut.print_c(f"[INFO] Fetching data for cluster: {cluster_name}; {n + 1}/{n_unique_cluster}")
            except IndexError:
                ut.print_c(f"[WARNING] Missing data for cluster {cluster_name}!")
                mean_expression = 0
            unique_cells_cluster_colors[cells_cluster_dataset == cluster_name] = mean_expression

        if len(unique_cells_cluster_colors) > 0:
            min_value = min(unique_cells_cluster_colors)
            #min_value = 0.0
            max_value = max(unique_cells_cluster_colors)
            #max_value = 8.0
            unique_cells_cluster_colors_norm = [(x - min_value) / (max_value - min_value) for x in
                                                unique_cells_cluster_colors]
            unique_cells_cluster_colors_norm = np.array(unique_cells_cluster_colors_norm)

        # unique_cells_cluster_colors_norm = unique_cells_cluster_colors_norm * 100
        # unique_cells_cluster_colors_norm = unique_cells_cluster_colors_norm * (2**16 - 1)
        # unique_cells_cluster_colors_norm = unique_cells_cluster_colors_norm.astype("uint16")


        voxelization_parameter = dict(
            shape=np.transpose(tifffile.imread(REFERENCE_FILE), (1, 2, 0)).shape,
            dtype="float",
            weights=unique_cells_cluster_colors_norm,
            method='sphere',
            radius=np.array([5, 5, 5]),
            kernel=None,
            processes=None,
            verbose=True,
            intensity="max",  # If None, perform count voxelization, otherwise 'mean' or 'max' intensity
        )

        if SHOW_GLIA:
            if len(cell_type) == 1:
                hm_path_all = os.path.join(SAVING_DIRECTORY, f"heatmap_{target_gene}_{cell_type[0]}.tif")
            else:
                hm_path_all = os.path.join(SAVING_DIRECTORY, f"heatmap_{target_gene}_glia.tif")
        else:
            hm_path_all = os.path.join(SAVING_DIRECTORY, f"heatmap_{target_gene}_neurons.tif")

        if os.path.exists(hm_path_all):
            os.remove(hm_path_all)
        hm = vox.voxelize(filtered_points_dataset,
                          **voxelization_parameter)

        cmap = matplotlib.colormaps["YlOrRd"]
        # cmap = matplotlib.colormaps["hot_r"]
        # n_colors = cmap.N  # Get the number of color entries in the colormap
        # half_cmap_data = cmap(np.linspace(0, 0.5, n_colors // 2))  # Get the first half
        # half_cmap_data = half_cmap_data[::-1]  # Flip the colormap data
        # half_cmap_data[0] = [1, 1, 1, 1]
        # new_cmap = LinearSegmentedColormap.from_list('half_flipped_cmap', half_cmap_data, N=n_colors)
        new_cmap = cmap

        # cmap = matplotlib.colormaps["YlOrBr"]
        # new_cmap = cmap

        # Step 1: Normalize the array to the range [0, 1]
        hm_min = hm.min()
        hm_max = hm.max()

        # Avoid division by zero in case data_min == data_max
        if hm_max > hm_min:
            normalized_hm = (hm - hm_min) / (hm_max - hm_min)
        else:
            normalized_hm = np.zeros_like(hm)
        # Apply the colormap to the normalized data
        # colormap expects 2D input, so flatten the data
        # REFERENCE_INV = (255 - REFERENCE)/4
        # flipped_ref = np.swapaxes(np.rot90(REFERENCE_INV, 1)[::-1], 1, 2)
        # normalized_ref = (flipped_ref - flipped_ref.min()) / (flipped_ref.max() - flipped_ref.min())
        # Create an array to hold the RGBA data with the new shape (512, 268, 369, 4)
        # rgba_ref = np.zeros((512, 268, 369, 4), dtype=float)
        # Fill the first three channels with the original data and add an alpha channel
        # for i in range(normalized_ref.shape[-1]):
            # Assume data[:, :, i] is grayscale or single-channel image data
            # rgba_ref[:, :, i, :3] = np.stack([normalized_ref[:, :, i]] * 3, axis=-1)  # Copy grayscale to RGB
            # rgba_ref[:, :, i, 3] = 1  # Set the alpha channel to 255 (fully opaque)
        colored_hm = new_cmap(normalized_hm)
        # colored_hm[:, :, int(colored_hm.shape[2]/2):] = rgba_ref[:, :, int(rgba_ref.shape[2]/2):]
        colored_hm[:, :, int(colored_hm.shape[2]/2):] = np.flip(colored_hm[:, :, :int(colored_hm.shape[2]/2)+1], 2)
        # colored_hm[:, :, int(colored_hm.shape[2]/2):] = 0
        tifffile.imwrite(hm_path_all, colored_hm)
