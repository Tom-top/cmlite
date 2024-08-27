import os
import json
import requests
import numpy as np
import pandas as pd
import tifffile
import anndata
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut

from spatial_transcriptomics.utils.coordinate_manipulation import filter_points_in_3d_mask

ATLAS_USED = "gubra"
DATASETS = np.arange(1, 6, 1)
N_DATASETS = len(DATASETS)
CATEGORY_NAMES = ["cluster"]
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]
target_genes = ["Calca", "Hcrtr1", "Hcrtr2", "Sod1"]
target_genes = ["Calca"]

DOWNLOAD_BASE = r"/default/path"  # PERSONAL
MAP_DIR = fr"/default/path"  # PERSONAL
TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"whole_brain_mask.tif"))
RESULTS_DIR = ut.create_dir(os.path.join(MAP_DIR, "results"))
SLICES_DIR = ut.create_dir(os.path.join(RESULTS_DIR, "thick_slices"))

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
mean_expression_matrix_path = r"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results\3d_views\cluster_log2_mean_gene_expression_merge_3.feather"
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
colormap = plt.get_cmap("YlOrBr")
# colormap = plt.get_cmap("viridis")
# colormap = plt.get_cmap("RdBu")

CHUNK_SIZE = 10
print("")

for m, ccat in enumerate(CATEGORY_NAMES):

    chunks_start = np.arange(0, TISSUE_MASK.shape[0], CHUNK_SIZE)
    chunks_end = np.arange(CHUNK_SIZE, TISSUE_MASK.shape[0], CHUNK_SIZE)
    if chunks_end[-1] != TISSUE_MASK.shape[0]:
        chunks_end = np.append(chunks_end, TISSUE_MASK.shape[0])
    n_chunks = len(chunks_start)

    for n, (cs, ce) in enumerate(zip(chunks_start, chunks_end)):

        ut.print_c(f"[INFO] Processing chunk: {cs}:{ce}. {n}/{n_chunks}")
        chunk_mask = TISSUE_MASK.copy()
        chunk_mask[0:cs] = 0
        chunk_mask[ce:] = 0

        filtered_points_plane = []
        cell_categories_all = []

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
            cell_categories = filtered_metadata_views[ccat].tolist()

            non_neuronal_mask = np.array(
                [True if any([j in i for j in NON_NEURONAL_CELL_TYPES]) else False for i in cell_classes])

            if filtered_points.size > 0:
                filtered_points_plane.append(filtered_points[~non_neuronal_mask])
                cell_categories_all.append(np.array(cell_categories)[~non_neuronal_mask])
            else:
                filtered_points_plane.append(np.array([]))
                cell_categories_all.append(np.array([]))

        ############################################################################################################
        # SAVE FLYTHROUGH IN CORONAL
        ############################################################################################################

        # Filter out empty arrays
        non_empty_arrays = [arr for arr in filtered_points_plane if arr.size > 0]
        non_empty_categories = [arr for arr in cell_categories_all if len(arr) > 0]
        if non_empty_arrays:
            filtered_points_plane = np.concatenate(non_empty_arrays)
            filtered_categories = np.concatenate(non_empty_categories)
            sorted_z_indices = np.argsort(filtered_points_plane[:, 0])
        else:
            filtered_points_plane = np.array([])
            filtered_categories = np.array([])
            sorted_z_indices = np.array([])
        # filtered_points_colors_plane = np.concatenate(filtered_points_colors_plane)

        unique_cell_clusters = np.unique(filtered_categories, return_index=True)
        n_unique_cluster = len(unique_cell_clusters[0])

        for target_gene in target_genes:
            SAVING_DIR = ut.create_dir(os.path.join(SLICES_DIR, target_gene))
            unique_cells_cluster_colors = []

            for n, cluster_name in enumerate(unique_cell_clusters[0]):
                ut.print_c(f"[INFO] Fetching data for cluster: {cluster_name}; {n + 1}/{n_unique_cluster}")
                cluster_mask = mean_expression_matrix["cluster_name"] == cluster_name
                try:
                    mean_expression = np.array(mean_expression_matrix[cluster_mask][target_gene])
                    if len(mean_expression_matrix[cluster_mask]) == 0:
                        ut.print_c(f"[WARNING] Missing data for cluster {cluster_name}!")
                        mean_expression = np.array([0])
                except KeyError:
                    ut.print_c(f"[WARNING] {target_gene} not found in mean expression table!")
                    mean_expression = np.array([0])
                unique_cells_cluster_colors.append(mean_expression)
            unique_cells_cluster_colors = np.array(unique_cells_cluster_colors).flatten()

            # for n, cluster_name in enumerate(unique_cell_clusters[0]):
            #     ut.print_c(f"[INFO] Fetching data for cluster: {cluster_name}; {n + 1}/{n_unique_cluster}")
            #     # cluster_name = cells_cluster_merged[cluster]
            #     cluster_mask = exp["cluster"] == cluster_name  # Mask of the cells belonging to the cluster
            #
            #     # Fixme: This has to be fixed as only the 10Xv3 dataset is fetched (the largest). 10Xv2 and 10XMulti or omitted
            #     library_mask = exp["library_method"] == "10Xv3"  # Mask of the library data
            #
            #     # Combined mask of cells in the dataset that belong to the selected cluster
            #     cluster_and_library_mask = np.logical_and(cluster_mask, library_mask)
            #
            #     if np.sum(cluster_and_library_mask) > 0:
            #         # Fetch gene expression data from the selected cluster
            #         # cell_labels_in = exp[cluster_and_library_mask].index  # Cell labels in the selected cluster
            #         cell_labels_in = exp[cluster_and_library_mask].index  # Cell labels in the selected cluster
            #         adata_cluster_in = []  # Gene expression of cells in the selected cluster for each sub-dataset
            #
            #         # Loop over each sub-dataset (.h5ad files)
            #         for adata in adatas:
            #             mask_in = adata.obs.index.isin(
            #                 cell_labels_in)  # Creates mask for the cells in the selected cluster
            #             adata_cluster_in.append(adata[mask_in])  # Appends result
            #
            #         adata_cluster_in_filtered = [x for x in adata_cluster_in if
            #                                      len(x) > 0]  # Filters out empty anndata
            #         combined_adata_in = anndata.concat(adata_cluster_in_filtered,
            #                                            axis=0)  # Combine all the datasets
            #
            #         combined_data_in = combined_adata_in.to_df()  # Load the expression data into memory
            #
            #         gene_mask = genes["gene_symbol"] == target_gene
            #         gene_id = genes["gene_identifier"][gene_mask]
            #         mean_gene_expression = float(np.mean(combined_data_in[gene_id], axis=0))
            #         unique_cells_cluster_colors.append(mean_gene_expression)
            #     else:
            #         mean_gene_expression = 0
            #         unique_cells_cluster_colors.append(mean_gene_expression)
            if len(unique_cells_cluster_colors) > 0:
                min_value = min(unique_cells_cluster_colors)
                # max_value = max(unique_cells_cluster_colors)
                max_value = 8.0
                unique_cells_cluster_colors_norm = [(x - min_value) / (max_value - min_value) for x in
                                                    unique_cells_cluster_colors]

                # cells_cluster_merged_colors = np.zeros((np.array(cells_cluster_merged).shape[0], 3), dtype=int)
                cells_cluster_merged_colors = np.full_like(filtered_categories, "")
                alpha_values = np.zeros(len(filtered_categories))
                for cluster_name, cluster_color in zip(unique_cell_clusters[0], unique_cells_cluster_colors_norm):
                    cluster_mask = np.array([True if i == cluster_name else False for i in filtered_categories])
                    rgb_color = colormap(cluster_color)[:3]
                    rgb_color_8b = np.array([i * 255 for i in rgb_color]).astype(int)
                    alpha_values[cluster_mask] = cluster_color
                    cells_cluster_merged_colors[cluster_mask] = ut.rgb_to_hex(rgb_color_8b)
            else:
                cells_cluster_merged_colors = np.array([])
                alpha_values = np.array([])
            alpha_values = [i if i >= 0.5 else 0.2 for i in alpha_values]
            alpha_values = [1 if i >= 1 else i for i in alpha_values]
            alpha_values = np.array(alpha_values)

            ut.print_c(f"[INFO] plotting {len(filtered_points_plane)} cells")

            fig = plt.figure()
            ax = plt.subplot(111)

            if filtered_points_plane.size > 0:
                ax.scatter(filtered_points_plane[:, 2][sorted_z_indices],
                           filtered_points_plane[:, 1][sorted_z_indices],
                           c=cells_cluster_merged_colors[sorted_z_indices], s=1,
                           lw=0., edgecolors="black", alpha=alpha_values[sorted_z_indices])

            reference_plane = (cs + ce) / 2
            coronal_ref = np.rot90(np.max(REFERENCE[:, cs:ce, :], 1))[::-1]
            coronal_ref[:, :int(coronal_ref.shape[1]/2)] = 0
            ax.imshow(coronal_ref, cmap='gray_r', alpha=0.3)
            ax.set_xlim(0, 369)
            ax.set_ylim(0, 268)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(os.path.join(SAVING_DIR, f"{ccat}_{cs}-{ce}.png"), dpi=300)
            plt.close()
