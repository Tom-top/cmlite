"""
This script generates a voxel-wise heatmap of cell densities from the combined ABC atlas datasets.
A heatmap is generated for all cells (i.e. neurons + non-neurons) and neurons alone

Author: Thomas Topilko
"""

import os

import numpy as np
import tifffile
import pandas as pd

import utils.utils as ut
import spatial_transcriptomics.utils.utils as sut

import analysis.measurements.voxelization as vox

########################################################################################################################
# SET GLOBAL VARIABLES
########################################################################################################################

ATLAS_USED = "aba"
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]

########################################################################################################################
# SET PATHS
########################################################################################################################

DOWNLOAD_BASE = r'E:\tto\spatial_transcriptomics'  # PERSONAL
URL = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
MANIFEST = sut.fetch_manifest(URL)
DATASETS = np.arange(1, 5, 1)

RESOURCES_DIR = "resources"
ATLAS_DIR = os.path.join(RESOURCES_DIR, "atlas")
ATLAS_FILE = os.path.join(ATLAS_DIR, f"{ATLAS_USED}_annotation_mouse.tif")
REFERENCE_FILE = os.path.join(ATLAS_DIR, f"{ATLAS_USED}_reference_mouse.tif")
TRANSFORM_DIR = os.path.join(RESOURCES_DIR, "abc_atlas")

WORKING_DIRECTORY = ut.create_dir(r"E:\tto\spatial_transcriptomics_results\whole_brain\results\3d_views")
SAVING_DIRECTORY = ut.create_dir(os.path.join(WORKING_DIRECTORY, f"/default/path"))  # PERSONAL

########################################################################################################################
# LOAD ATLAS AND REFERENCE
########################################################################################################################

ATLAS = np.swapaxes(tifffile.imread(ATLAS_FILE), 0, 2)
REFERENCE = np.swapaxes(tifffile.imread(REFERENCE_FILE), 0, 2)

transformed_coordinates_all_cells = []
transformed_coordinates_neurons = []

for dataset_n in DATASETS:

    dataset_id = f"Zhuang-ABCA-{dataset_n}"
    metadata = MANIFEST['file_listing'][dataset_id]['metadata']
    metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
    metadata_ccf = MANIFEST['file_listing'][f'{dataset_id}-CCF']['metadata']

    # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
    cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
    cell_metadata_file_ccf = os.path.join(DOWNLOAD_BASE, cell_metadata_path_ccf)
    cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
    cell_metadata_ccf.set_index('cell_label', inplace=True)
    cell_labels = cell_metadata_ccf.index

    # Views
    cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
    cell_metadata_file_views = os.path.join(DOWNLOAD_BASE, cell_metadata_path_views)
    cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
    cell_metadata_views.set_index('cell_label', inplace=True)

    filtered_metadata_views = cell_metadata_views.loc[cell_labels]
    cells_class = filtered_metadata_views["class"].tolist()
    non_neuronal_mask = np.array(
        [True if any([j in i for j in NON_NEURONAL_CELL_TYPES]) else False for i in cells_class])

    transformed_coordinates = np.load(os.path.join(TRANSFORM_DIR,
                                                   f"all_transformed_cells_{ATLAS_USED}_{dataset_n}.npy"))

    transformed_coordinates_all_cells.append(transformed_coordinates)
    transformed_coordinates_neurons.append(transformed_coordinates[~non_neuronal_mask])

transformed_coordinates_all_stack = np.vstack(transformed_coordinates_all_cells)
transformed_coordinates_neurons_stack = np.vstack(transformed_coordinates_neurons)

voxelization_parameter = dict(
    shape=np.transpose(tifffile.imread(REFERENCE_FILE), (1, 2, 0)).shape,
    dtype="uint16",
    weights=None,
    method='sphere',
    radius=np.array([1, 1, 1]),
    kernel=None,
    processes=None,
    verbose=True
)

hm_path_all = os.path.join(SAVING_DIRECTORY, f"heatmap_all_sphere_{ATLAS_USED}.tif")
if os.path.exists(hm_path_all):
    os.remove(hm_path_all)
hm = vox.voxelize(transformed_coordinates_all_stack,
                  **voxelization_parameter)
tifffile.imwrite(hm_path_all, hm)

hm_path_neurons = os.path.join(SAVING_DIRECTORY, f"heatmap_neurons_sphere_{ATLAS_USED}.tif")
if os.path.exists(hm_path_neurons):
    os.remove(hm_path_neurons)
hm = vox.voxelize(transformed_coordinates_neurons_stack,
                  **voxelization_parameter)
tifffile.imwrite(hm_path_neurons, hm)
