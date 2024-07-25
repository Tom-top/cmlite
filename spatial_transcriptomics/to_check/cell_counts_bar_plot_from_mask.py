"""
This script generates visualizations showing the distribution of the cells within the combined datasets
in a restricted binary mask. For each dataset, coronal, sagittal, and horizontal views are produced with
the Gubra LSFM reference in the background. The cells are labeled according to the ABC atlas ontology
(https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas). The source of the data is from Zhang M. et al.,
in Nature, 2023 (DOI: 10.1038/s41586-023-06808-9).

Author: Thomas Topilko
"""

import os
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")

from spatial_transcriptomics.utils.coordinate_manipulation import filter_points_in_3d_mask
from spatial_transcriptomics.utils.plotting import bar_plot
from spatial_transcriptomics.utils.utils import fetch_manifest, load_data, load_csv

# Constants
datasets = [1, 2, 3, 4]
DOWNLOAD_BASE = r"E:\tto\spatial_transcriptomics"
MAP_DIR = r"E:\tto\spatial_transcriptomics_results\Semaglutide"
TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"whole_brain_mask.tif"))
RESULTS_DIR = os.path.join(MAP_DIR, "results")
TRANSFORM_DIR = r"resources/abc_atlas"
REFERENCE_FILE = r"resources/atlas/gubra_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen"]
MANIFEST_URL = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'

# Helper functions

def get_unique_colors(metadata, category, color_category, non_neuronal_types):
    cells = metadata[category].tolist()
    colors = metadata[color_category].tolist()
    unique_cells, unique_indices = np.unique(cells, return_index=True)
    unique_colors = np.array(colors)[unique_indices]

    neuronal_mask_cells = np.array([True if any([j in i for j in non_neuronal_types]) else False for i in cells])
    neuronal_mask_categories = np.array([True if any([j in i for j in non_neuronal_types]) else False for i in unique_cells])

    return (np.array(colors)[~neuronal_mask_cells],
            unique_cells[~neuronal_mask_categories],
            unique_colors[~neuronal_mask_categories])

def process_dataset(dataset_n, manifest, download_base, tissue_mask, transform_dir):
    print(f"Fetching data from mouse {dataset_n}!")

    dataset_id = f"Zhuang-ABCA-{dataset_n}"
    metadata = manifest['file_listing'][dataset_id]['metadata']
    metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
    metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']
    expression_matrices = manifest['file_listing'][dataset_id]['expression_matrices']

    cell_metadata_path = os.path.join(download_base, expression_matrices[dataset_id]['log2']['files']['h5ad']['relative_path'])
    adata = load_data(cell_metadata_path)

    cell_metadata_path_ccf = os.path.join(download_base, metadata_ccf['ccf_coordinates']['files']['csv']['relative_path'])
    cell_metadata_ccf = load_csv(cell_metadata_path_ccf)

    cell_labels = cell_metadata_ccf.index
    transformed_coordinates = np.load(os.path.join(transform_dir, f"all_transformed_cells_{dataset_n}.npy"))

    cell_metadata_path_views = os.path.join(download_base, metadata_with_clusters['files']['csv']['relative_path'])
    cell_metadata_views = load_csv(cell_metadata_path_views)

    filtered_points, mask_point = filter_points_in_3d_mask(transformed_coordinates, tissue_mask)
    filtered_labels = np.array(cell_labels)[::-1][mask_point]
    filtered_metadata_views = cell_metadata_views.loc[filtered_labels]

    categories = ['class', 'subclass', 'supertype', 'cluster']
    color_categories = ['class_color', 'subclass_color', 'supertype_color', 'cluster_color']

    data = {}
    for category, color_category in zip(categories, color_categories):
        cells_colors, unique_cells, unique_cells_colors = get_unique_colors(filtered_metadata_views, category, color_category, NON_NEURONAL_CELL_TYPES)
        data[f'{category}_colors'] = cells_colors
        data[f'unique_{category}'] = unique_cells
        data[f'unique_{category}_colors'] = unique_cells_colors

    return data

def save_bar_plots(data, results_dir):
    categories = ['class', 'subclass', 'supertype', 'cluster']
    for category in categories:
        bar_plot(data[f'{category}_colors'],
                 data[f'unique_{category}'],
                 data[f'unique_{category}_colors'],
                 os.path.join(results_dir, f"cell_counts_{category}.png"))

# Main script
manifest = fetch_manifest(MANIFEST_URL)
aggregated_data = {f'{category}_colors': [] for category in ['class', 'subclass', 'supertype', 'cluster']}
aggregated_data.update({f'unique_{category}': [] for category in ['class', 'subclass', 'supertype', 'cluster']})
aggregated_data.update({f'unique_{category}_colors': [] for category in ['class', 'subclass', 'supertype', 'cluster']})

for dataset_n in datasets:
    data = process_dataset(dataset_n, manifest, DOWNLOAD_BASE, TISSUE_MASK, TRANSFORM_DIR)
    for key in aggregated_data.keys():
        aggregated_data[key].append(data[key])

for key in aggregated_data.keys():
    if key.startswith("unique"):
        aggregated_data[key] = np.unique(np.concatenate(aggregated_data[key]))
    else:
        aggregated_data[key] = np.concatenate(aggregated_data[key])

save_bar_plots(aggregated_data, RESULTS_DIR)
