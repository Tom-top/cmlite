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
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

import utils.utils as ut

from spatial_transcriptomics.utils.coordinate_manipulation import filter_points_in_3d_mask
import spatial_transcriptomics.utils.plotting as st_plt

ATLAS_USED = "gubra"
DATASETS = np.arange(1, 6, 1)
N_DATASETS = len(DATASETS)
CATEGORY_NAMES = ["neurotransmitter", "class", "subclass", "supertype", "cluster"]
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]
BILATERAL = True  # If True: generate bilateral cell distribution in the 3D representations
ONLY_NEURONS = True  # If True: only generate plots for neurons, excluding all non-neuronal cells
PERCENTAGE_THRESHOLD = 50
categories = ["cluster"]
target_genes = ["Slc17a6"]

ANO_DIRECTORY = r"resources\atlas"
ANO_PATH = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.tif")
ANO = np.transpose(tifffile.imread(ANO_PATH), (1, 2, 0))
ANO_JSON = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.json")

DOWNLOAD_BASE = r"/default/path"  # PERSONAL
MAP_DIR = ut.create_dir(rf"/default/path")  # PERSONAL
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
exp = pd.read_csv(metadata_file)  # Load metadata
exp.set_index('cell_label', inplace=True)  # Set cell_label as dataframe index

metadata_genes_relative_path = metadata_json['gene']['files']['csv']['relative_path']
metadata_gene_file = os.path.join(DOWNLOAD_BASE, metadata_genes_relative_path)  # Path to metadata file
genes = pd.read_csv(metadata_gene_file)  # Load metadata

# Fixme: This should be fixed as only the 10Xv3 dataset is fetched (the largest). 10Xv2 and 10XMulti or omitted
dataset_id = "WMB-10Xv3"  # Dataset name
metadata_exp = manifest['file_listing'][dataset_id]['expression_matrices']
adatas = []
for dsn in metadata_exp:
    print("")
    ut.print_c(f"[INFO] Loading 10x data for: {dsn}")
    adata = anndata.read_h5ad(os.path.join(DOWNLOAD_BASE,
                                           metadata_exp[dsn]["log2"]["files"]["h5ad"]["relative_path"]),
                              backed='r')
    adatas.append(adata)

print("")
ut.print_c("[INFO] Loading mean gene expression matrix!")
mean_expression_matrix_path = r"resources\abc_atlas\cluster_log2_mean_gene_expression_merge.feather"
mean_expression_matrix = pd.read_feather(mean_expression_matrix_path)

# colormap = plt.get_cmap("YlOrRd")
colormap = plt.get_cmap("gist_heat_r")

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

        ut.print_c(f"[INFO] Loading dataset: {dataset_n}")

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

    for target_gene in target_genes:
        unique_cells_cluster_colors = []

        for n, cluster_name in enumerate(unique_cells_cluster[0]):
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

        np.save(os.path.join(SAVING_DIR, f"unique_cells_cluster_colors_{target_gene}.npy"), unique_cells_cluster_colors)


        # NORMALIZE
        unique_cells_cluster_colors = [2**i for i in unique_cells_cluster_colors]
        unique_cells_cluster_colors = np.array(unique_cells_cluster_colors)
        unique_cells_cluster_colors[unique_cells_cluster_colors >= 1500] = 1500
        # min_value = min(unique_cells_cluster_colors)
        min_value = 0
        # min_value = 0
        # max_value = max(unique_cells_cluster_colors)
        max_value = 1500
        # max_value = 8
        unique_cells_cluster_colors_norm = [(x - min_value) / (max_value - min_value) for x in unique_cells_cluster_colors]

        # cells_cluster_merged_colors = np.full_like(cells_cluster_merged, "")
        # for m, (cluster_name, cluster_color) in enumerate(zip(unique_cells_cluster[0], unique_cells_cluster_colors_norm)):
        #     ut.print_c(f"[INFO] Applying colormap for cluster: {cluster_name}; {m + 1}/{n_unique_cluster}")
        #     cluster_mask = np.array([True if i == cluster_name else False for i in cells_cluster_merged])
        #     rgb_color = colormap(cluster_color)[:3]
        #     rgb_color_8b = np.array([i*255 for i in rgb_color]).astype(int)
        #     cells_cluster_merged_colors[cluster_mask] = ut.rgb_to_hex(rgb_color_8b)

        cells_cluster_merged_colors = np.full_like(cells_cluster_merged, "")
        cells_cluster_merged_arr = np.array(cells_cluster_merged)
        # Precompute RGB colors in 8-bit format for all clusters at once
        rgb_colors_8b = np.array(
            [np.array(colormap(cluster_color)[:3]) * 255 for cluster_color in unique_cells_cluster_colors_norm]).astype(
            int)
        # Precompute hex colors from the RGB 8-bit values
        hex_colors = np.array([ut.rgb_to_hex(rgb_color) for rgb_color in rgb_colors_8b])
        # Vectorized assignment of colors
        for m, cluster_name in enumerate(unique_cells_cluster[0]):
            ut.print_c(f"[INFO] Applying colormap for cluster: {cluster_name}; {m + 1}/{n_unique_cluster}")
            # Apply mask and assign hex colors
            cells_cluster_merged_colors[cells_cluster_merged_arr == str(cluster_name)] = hex_colors[m]

        ########################################################################################################
        # PLOT CELLS IN 3D
        ########################################################################################################

        filtered_points_merged_conc = np.array([])
        filtered_points_merged = np.array(filtered_points_merged)
        cell_size = 0.5
        cell_size_global = 0.5

        if BILATERAL:
            mirrored_filtered_points = filtered_points_merged.copy()
            if mirrored_filtered_points.size > 0:
                mirrored_filtered_points[:, 2] = REFERENCE.shape[0] - 1 - filtered_points_merged[:, 2]
            if filtered_points_merged.shape[0] != filtered_points_merged_conc.shape[0]:
                filtered_points_merged_conc = np.concatenate([filtered_points_merged, mirrored_filtered_points])
            if non_neuronal_mask_global.shape[0] != filtered_points_merged_conc.shape[0]:
                non_neuronal_mask_global = np.tile(non_neuronal_mask_global, 2)

        for cat in categories:

            if BILATERAL:
                points_cats = np.tile(np.array(globals()[f"cells_{cat}_merged"]), 2)
                points_colors = np.tile(cells_cluster_merged_colors, 2)
            else:
                points_cats = np.array(globals()[f"cells_{cat}_merged"])
                points_colors = cells_cluster_merged_colors

            ########################################################################################################
            # HORIZONTAL 3D VIEW
            ########################################################################################################
            ori = "horizontal"
            orix, oriy, mask_axis = 2, 0, 1
            xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[1]

            if not ONLY_NEURONS:
                # All cells, class colors, all experiments
                st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                                  cell_categories=points_cats, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                                  orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                                  sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"all_{ori}_mouse_{cat}_{target_gene}.png"),
                                  relevant_categories=[], show_outline=SHOW_OUTLINE, zoom=ZOOM,
                                  show_ref=SHOW_REF, surface_projection=False)
            # Only neurons, class colors, all experiments
            st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                              cell_categories=points_cats, non_neuronal_mask=non_neuronal_mask_global, xlim=xlim,
                              ylim=ylim,
                              orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                              sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"neurons_{ori}_mouse_{cat}_{target_gene}.png"),
                              relevant_categories=[], show_outline=SHOW_OUTLINE, zoom=ZOOM,
                              show_ref=SHOW_REF, surface_projection=False)

            ########################################################################################################
            # SAGITTAL 3D VIEW
            ########################################################################################################

            ori = "sagittal"
            orix, oriy, mask_axis = 0, 1, 2
            xlim, ylim = REFERENCE_SHAPE[1], REFERENCE_SHAPE[2]

            if not ONLY_NEURONS:
                # All cells, class colors, all experiments
                st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                                  cell_categories=points_cats, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                                  orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                                  sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"all_{ori}_mouse_{cat}_{target_gene}.png"),
                                  relevant_categories=[], show_outline=SHOW_OUTLINE, zoom=ZOOM,
                                  show_ref=SHOW_REF, surface_projection=False)
            # Only neurons, class colors, all experiments
            st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                              cell_categories=points_cats, non_neuronal_mask=non_neuronal_mask_global, xlim=xlim, ylim=ylim,
                              orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                              sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"neurons_{ori}_mouse_{cat}_{target_gene}.png"),
                              relevant_categories=[], show_outline=SHOW_OUTLINE, zoom=ZOOM,
                              show_ref=SHOW_REF, surface_projection=False)

            ########################################################################################################
            # CORONAL 3D VIEW
            ########################################################################################################

            ori = "coronal"
            orix, oriy, mask_axis = 2, 1, 0  # Projection = 1
            xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[2]

            if not ONLY_NEURONS:
                # All cells, class colors, all experiments
                st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                                  cell_categories=points_cats, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                                  orix=orix, oriy=oriy, orip=oriy, ori=ori, mask_axis=mask_axis, s=cell_size,
                                  sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"all_{ori}_mouse_{cat}_{target_gene}.png"),
                                  relevant_categories=[], show_outline=SHOW_OUTLINE, zoom=ZOOM,
                                  show_ref=SHOW_REF, surface_projection=False)
            # Only neurons, class colors, all experiments
            st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                              cell_categories=points_cats, non_neuronal_mask=non_neuronal_mask_global, xlim=xlim,
                              ylim=ylim,
                              orix=orix, oriy=oriy, orip=oriy, ori=ori, mask_axis=mask_axis, s=cell_size,
                              sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"neurons_{ori}_mouse_{cat}_{target_gene}.png"),
                              relevant_categories=[], show_outline=SHOW_OUTLINE, zoom=ZOOM,
                              show_ref=SHOW_REF, surface_projection=False)
