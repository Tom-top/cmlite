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
NON_NEURONAL_CELL_TYPES = [["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]]
# NON_NEURONAL_CELL_TYPES = [["Astro"], ["Oligo"], ["Vascular"], ["Immune"], ["Epen"], ["OEC"]]
SHOW_GLIA = False
target_genes = ["Glp1r", "Gip", "Ramp1", "Ramp2", "Ramp3", "Calca"]
normalization_max_value = 8.0
CHUNK_SIZE = 10

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
mean_expression_matrix_path = r"resources\abc_atlas\cluster_log2_mean_gene_expression_merge.feather"
mean_expression_matrix = pd.read_feather(mean_expression_matrix_path)

########################################################################################################################
# TEST
#
# clusters = ['4601 HB Calcb Chol_1', '2761 RN Spp1 Glut_1']  # 10xRSeq_Mult
# clusters = ['0120 L2/3 IT CTX Glut_4']  # 10Xv2
#
# print(np.where(np.isnan(unique_cells_cluster_colors)))
# print(unique_cells_cluster_colors[757])
# print(unique_cell_clusters[0][112])
#
# print(exp["cluster"][exp["cluster"] == unique_cell_clusters[0][1560]])
# print(exp["library_method"][exp["cluster"] == unique_cell_clusters[0][112]])
# print(np.sum(exp["library_method"][exp["cluster"] == unique_cell_clusters[0][1560]] == "10Xv3"))
#
# adatas = []
# print("")
# for dsn in metadata_exp:
#     ut.print_c(f"[INFO] Loading 10x data for: {dsn}")
#     adata = anndata.read_h5ad(os.path.join(DOWNLOAD_BASE,
#                                            metadata_exp[dsn]["log2"]["files"]["h5ad"]["relative_path"]),
#                               backed='r')
#     adatas.append(adata)

########################################################################################################################

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

print("")

for m, ccat in enumerate(CATEGORY_NAMES):

    chunks_start = np.arange(0, TISSUE_MASK.shape[0], CHUNK_SIZE)
    chunks_end = np.arange(CHUNK_SIZE, TISSUE_MASK.shape[0], CHUNK_SIZE)
    if chunks_end[-1] != TISSUE_MASK.shape[0]:
        chunks_end = np.append(chunks_end, TISSUE_MASK.shape[0])
    n_chunks = len(chunks_start)

    for cell_type in NON_NEURONAL_CELL_TYPES:

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

                if SHOW_GLIA:
                    non_neuronal_mask = np.array(
                        [False if any([j in i for j in cell_type]) else True for i in cell_classes])
                else:
                    non_neuronal_mask = np.array(
                        [True if any([j in i for j in cell_type]) else False for i in cell_classes])

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
                gene_dir = ut.create_dir(os.path.join(SLICES_DIR, target_gene))
                unique_cells_cluster_colors = []
                min_gene_expression = np.min(mean_expression_matrix[target_gene])
                max_gene_expression = np.max(mean_expression_matrix[target_gene])
                np.save(os.path.join(gene_dir, "min_max_log2_CPM.npy"),
                        np.array([min_gene_expression, max_gene_expression]))

                for n, cluster_name in enumerate(unique_cell_clusters[0]):
                    ut.print_c(f"[INFO] Fetching data for cluster: {cluster_name}; {n + 1}/{n_unique_cluster}")
                    cluster_mask = mean_expression_matrix["cluster_name"] == cluster_name
                    try:
                        mean_expression = np.array(mean_expression_matrix[cluster_mask][target_gene])
                        if np.isnan(mean_expression):  # Fixme: This scips clusters
                            # Fixme: ['4601 HB Calcb Chol_1', '2761 RN Spp1 Glut_1', '0120 L2/3 IT CTX Glut_4'] as they are not in the 10Xv3 datasets
                            mean_expression = np.array([0])
                        if len(mean_expression_matrix[cluster_mask]) == 0:
                            ut.print_c(f"[WARNING] Missing data for cluster {cluster_name}!")
                            mean_expression = np.array([0])
                    except KeyError:
                        ut.print_c(f"[WARNING] {target_gene} not found in mean expression table!")
                        mean_expression = np.array([0])
                    unique_cells_cluster_colors.append(mean_expression)
                unique_cells_cluster_colors = np.array(unique_cells_cluster_colors).flatten()
                for m in range(2):
                    if len(unique_cells_cluster_colors) > 0:
                        if m == 0:
                            min_value = 0
                            max_value = normalization_max_value
                            saving_text = "manual_norm"
                        elif m == 1:
                            min_value = min_gene_expression
                            max_value = max_gene_expression
                            saving_text = "min_max_norm"
                        unique_cells_cluster_colors_norm = [(x - min_value) / (max_value - min_value) for x in
                                                            unique_cells_cluster_colors]

                        # cells_cluster_merged_colors = np.zeros((np.array(cells_cluster_merged).shape[0], 3), dtype=int)
                        cells_cluster_merged_colors = np.full_like(filtered_categories, "")
                        alpha_values = np.zeros(len(filtered_categories))
                        for cluster_name, cluster_color in zip(unique_cell_clusters[0],
                                                               unique_cells_cluster_colors_norm):
                            cluster_mask = np.array([True if i == cluster_name else False for i in filtered_categories])
                            rgb_color = colormap(cluster_color)[:3]
                            rgb_color_8b = np.array([i * 255 for i in rgb_color]).astype(int)
                            alpha_values[cluster_mask] = cluster_color
                            cells_cluster_merged_colors[cluster_mask] = ut.rgb_to_hex(rgb_color_8b)
                    else:
                        if m == 0:
                            saving_text = "manual_norm"
                        elif m == 1:
                            saving_text = "min_max_norm"
                        cells_cluster_merged_colors = np.array([])
                        alpha_values = np.array([])
                    alpha_values = [i if i >= 0.5 else 0.2 for i in alpha_values]
                    alpha_values = [1 if i >= 1 else i for i in alpha_values]
                    alpha_values = np.array(alpha_values)

                    if SHOW_GLIA:
                        if len(cell_type) == 1:
                            SAVING_DIR = ut.create_dir(os.path.join(gene_dir, target_gene + f"_{cell_type[0]}_{saving_text}"))
                        else:
                            SAVING_DIR = ut.create_dir(os.path.join(gene_dir, target_gene + f"_glia_{saving_text}"))
                    else:
                        SAVING_DIR = ut.create_dir(os.path.join(gene_dir, target_gene + f"_neurons_{saving_text}"))

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
                    coronal_ref[:, :int(coronal_ref.shape[1] / 2)] = 0
                    ax.imshow(coronal_ref, cmap='gray_r', alpha=0.3)
                    ax.set_xlim(0, 369)
                    ax.set_ylim(0, 268)
                    ax.invert_yaxis()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if m == 0:
                        plt.savefig(os.path.join(SAVING_DIR, f"{ccat}_{cs}-{ce}.png"), dpi=300)
                    else:
                        plt.savefig(os.path.join(SAVING_DIR, f"{ccat}_{cs}-{ce}.png"), dpi=300)
                    plt.close()
