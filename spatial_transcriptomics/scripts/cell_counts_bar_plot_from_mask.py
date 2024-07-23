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
import anndata
import matplotlib
matplotlib.use("Agg")

from spatial_transcriptomics.utils.coordinate_manipulation import filter_points_in_3d_mask
from spatial_transcriptomics.utils.plotting import bar_plot

datasets = [1, 2, 3, 4]
n_datasets = len(datasets)

DOWNLOAD_BASE = r"E:\tto\spatial_transcriptomics"  # PERSONAL
MAP_DIR = r"E:\tto\spatial_transcriptomics_results\Semaglutide"  # PERSONAL
TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"whole_brain_mask.tif"))
RESULTS_DIR = os.path.join(MAP_DIR, "results")

TRANSFORM_DIR = r"resources/abc_atlas"
REFERENCE_FILE = r"resources/atlas/gubra_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)

cells_cls_colors = []
unique_cells_classes = []
unique_cells_cls_colors = []

cells_subcls_colors = []
unique_cells_subclasses = []
unique_cells_subcls_colors = []

cells_supertype_colors = []
unique_cells_supertypes = []
unique_cells_supertype_colors = []

cells_cluster_colors = []
unique_cells_clusters = []
unique_cells_cluster_colors = []

for i, dataset_n in enumerate(datasets):

    print(f"Fetching data from mouse {dataset_n}!")

    dataset_id = f"Zhuang-ABCA-{dataset_n}"
    url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
    manifest = json.loads(requests.get(url).text)
    metadata = manifest['file_listing'][dataset_id]['metadata']
    metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
    metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']
    expression_matrices = manifest['file_listing'][dataset_id]['expression_matrices']
    cell_metadata_path = expression_matrices[dataset_id]['log2']['files']['h5ad']['relative_path']
    file = os.path.join(DOWNLOAD_BASE, cell_metadata_path)
    adata = anndata.read_h5ad(file, backed='r')
    genes = adata.var

    # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
    cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
    cell_metadata_file_ccf = os.path.join(DOWNLOAD_BASE, cell_metadata_path_ccf)
    cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
    cell_metadata_ccf.set_index('cell_label', inplace=True)
    cell_labels = cell_metadata_ccf.index
    n_cells_ccf = len(cell_metadata_ccf)

    # Filter out the cells
    transformed_coordinates = np.load(os.path.join(TRANSFORM_DIR, f"all_transformed_cells_{dataset_n}.npy"))

    # Views
    cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
    cell_metadata_file_views = os.path.join(DOWNLOAD_BASE, cell_metadata_path_views)
    cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
    cell_metadata_views.set_index('cell_label', inplace=True)

    filtered_points, mask_point = filter_points_in_3d_mask(transformed_coordinates, TISSUE_MASK)
    filtered_labels = np.array(cell_labels)[::-1][mask_point]
    filtered_metadata_views = cell_metadata_views.loc[filtered_labels]

    # Extract data for each category
    cells_class = filtered_metadata_views["class"].tolist()
    cells_cls_color = filtered_metadata_views["class_color"].tolist()
    cells_subclass = filtered_metadata_views["subclass"].tolist()
    cells_subcls_color = filtered_metadata_views["subclass_color"].tolist()
    cells_supertype = filtered_metadata_views["supertype"].tolist()
    cells_supertype_color = filtered_metadata_views["supertype_color"].tolist()
    cells_cluster = filtered_metadata_views["cluster"].tolist()
    cells_cluster_color = filtered_metadata_views["cluster_color"].tolist()

    unique_cells_class, unique_indices = np.unique(cells_class, return_index=True)
    unique_cells_cls_color = np.array(cells_cls_color)[unique_indices]

    unique_cells_subclass, unique_indices = np.unique(cells_subclass, return_index=True)
    unique_cells_subcls_color = np.array(cells_subcls_color)[unique_indices]

    unique_cells_supertype, unique_indices = np.unique(cells_supertype, return_index=True)
    unique_cells_supertype_color = np.array(cells_supertype_color)[unique_indices]

    unique_cells_cluster, unique_indices = np.unique(cells_cluster, return_index=True)
    unique_cells_cluster_color = np.array(cells_cluster_color)[unique_indices]

    non_neuronal_cell_types = ["Astro", "Oligo", "Vascular", "Immune", "Epen"]

    neuronal_mask_cells = np.array(
        [True if any([j in i for j in non_neuronal_cell_types]) else False for i in cells_class])
    neuronal_mask_categories = np.array(
        [True if any([j in i for j in non_neuronal_cell_types]) else False for i in unique_cells_class])

    cells_cls_colors.append(np.array(cells_cls_color)[~neuronal_mask_cells])
    unique_cells_classes.append(np.array(unique_cells_class)[~neuronal_mask_categories])
    unique_cells_cls_colors.append(np.array(unique_cells_cls_color)[~neuronal_mask_categories])

    neuronal_mask_cells = np.array(
        [True if any([j in i for j in non_neuronal_cell_types]) else False for i in cells_subclass])
    neuronal_mask_categories = np.array(
        [True if any([j in i for j in non_neuronal_cell_types]) else False for i in unique_cells_subclass])

    cells_subcls_colors.append(np.array(cells_subcls_color)[~neuronal_mask_cells])
    unique_cells_subclasses.append(np.array(unique_cells_subclass)[~neuronal_mask_categories])
    unique_cells_subcls_colors.append(np.array(unique_cells_subcls_color)[~neuronal_mask_categories])

    neuronal_mask_cells = np.array(
        [True if any([j in i for j in non_neuronal_cell_types]) else False for i in cells_supertype])
    neuronal_mask_categories = np.array(
        [True if any([j in i for j in non_neuronal_cell_types]) else False for i in unique_cells_supertype])

    cells_supertype_colors.append(np.array(cells_supertype_color)[~neuronal_mask_cells])
    unique_cells_supertypes.append(np.array(unique_cells_supertype)[~neuronal_mask_categories])
    unique_cells_supertype_colors.append(np.array(unique_cells_supertype_color)[~neuronal_mask_categories])

    neuronal_mask_cells = np.array(
        [True if any([j in i for j in non_neuronal_cell_types]) else False for i in cells_cluster])
    neuronal_mask_categories = np.array(
        [True if any([j in i for j in non_neuronal_cell_types]) else False for i in unique_cells_cluster])

    cells_cluster_colors.append(np.array(cells_cluster_color)[~neuronal_mask_cells])
    unique_cells_clusters.append(np.array(unique_cells_cluster)[~neuronal_mask_categories])
    unique_cells_cluster_colors.append(np.array(unique_cells_cluster_color)[~neuronal_mask_categories])

cells_cls_colors = np.concatenate(cells_cls_colors)
unique_cells_classes = np.unique(np.concatenate(unique_cells_classes))
unique_cells_cls_colors = np.unique(np.concatenate(unique_cells_cls_colors))
bar_plot(cells_cls_colors, unique_cells_classes, unique_cells_cls_colors,
         os.path.join(RESULTS_DIR, f"cell_counts_class.png"))

cells_subcls_colors = np.concatenate(cells_subcls_colors)
unique_cells_subclasses = np.unique(np.concatenate(unique_cells_subclasses))
unique_cells_subcls_colors = np.unique(np.concatenate(unique_cells_subcls_colors))
bar_plot(cells_subcls_colors, unique_cells_subclasses, unique_cells_subcls_colors,
         os.path.join(RESULTS_DIR, f"cell_counts_subclass.png"))

cells_supertype_colors = np.concatenate(cells_supertype_colors)
unique_cells_supertypes = np.unique(np.concatenate(unique_cells_supertypes))
unique_cells_supertype_colors = np.unique(np.concatenate(unique_cells_supertype_colors))
bar_plot(cells_supertype_colors, unique_cells_supertypes, unique_cells_supertype_colors,
         os.path.join(RESULTS_DIR, f"cell_counts_supertype.png"))

cells_cluster_colors = np.concatenate(cells_cluster_colors)
unique_cells_clusters = np.unique(np.concatenate(unique_cells_clusters))
unique_cells_cluster_colors = np.unique(np.concatenate(unique_cells_cluster_colors))
bar_plot(cells_cluster_colors, unique_cells_clusters, unique_cells_cluster_colors,
         os.path.join(RESULTS_DIR, f"cell_counts_cluster.png"))