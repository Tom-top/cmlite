"""
This script generates a region-wise heatmap of cell densities from the combined ABC atlas datasets.
A heatmap is generated for all cells (i.e. neurons + non-neurons) and neurons alone

Author: Thomas Topilko
"""

import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut
import spatial_transcriptomics.utils.utils as sut
import spatial_transcriptomics.utils.coordinate_manipulation as scm

########################################################################################################################
# SET GLOBAL VARIABLES
########################################################################################################################

ATLAS_USED = "gubra"
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]

########################################################################################################################
# SET PATHS
########################################################################################################################

DOWNLOAD_BASE = r'E:\tto\spatial_transcriptomics'  # PERSONAL
URL = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
MANIFEST = sut.fetch_manifest(URL)
DATASETS = np.arange(1, 6, 1)

RESOURCES_DIR = "resources"
ATLAS_DIR = os.path.join(RESOURCES_DIR, "atlas")
ATLAS_FILE = os.path.join(ATLAS_DIR, f"{ATLAS_USED}_annotation_mouse.tif")
REFERENCE_FILE = os.path.join(ATLAS_DIR, f"{ATLAS_USED}_reference_mouse.tif")
TRANSFORM_DIR = os.path.join(RESOURCES_DIR, "abc_atlas")

SAVING_DIRECTORY = ut.create_dir(fr"/default/path")  # PERSONAL
SAVING_DIRECTORY_SLICES = ut.create_dir(os.path.join(SAVING_DIRECTORY, "neuron_density_coronal"))

########################################################################################################################
# LOAD ATLAS AND REFERENCE
########################################################################################################################

ATLAS = np.swapaxes(tifffile.imread(ATLAS_FILE), 0, 2)
ATLAS_SHAPE = ATLAS.shape
REFERENCE = np.swapaxes(tifffile.imread(REFERENCE_FILE), 0, 2)

########################################################################################################################
# CLIP ATLAS & REFERENCE
########################################################################################################################

CLIPPED_ATLAS = ATLAS.copy()
CLIPPED_ATLAS[:, :, int(np.ceil(ATLAS_SHAPE[-1] / 2)):] = 0
CLIPPED_ATLAS_MASK = CLIPPED_ATLAS > 0
CLIPPED_REFERENCE = REFERENCE.copy()
CLIPPED_REFERENCE[:, :, int(np.ceil(ATLAS_SHAPE[-1] / 2)):] = 0

########################################################################################################################
# LOOP OVER EVERY ATLAS ID AND GET THE CELL COUNT IN THE REGION MASK
########################################################################################################################

unique_atlas_values = np.unique(CLIPPED_ATLAS)
n_unique_atlas_values = len(unique_atlas_values)
region_cells_counts = {}
region_neuronal_counts = {}
region_voxel_size = {}

# dummy = np.zeros_like(CLIPPED_ATLAS)

for dataset_n in DATASETS:

    if dataset_n < 5:
        dataset_id = f"Zhuang-ABCA-{dataset_n}"
    else:
        dataset_id = f"MERFISH-C57BL6J-638850"
    metadata = MANIFEST['file_listing'][dataset_id]['metadata']
    metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
    metadata_ccf = MANIFEST['file_listing'][f'{dataset_id}-CCF']['metadata']

    # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
    cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
    cell_metadata_file_ccf = os.path.join(DOWNLOAD_BASE, cell_metadata_path_ccf)
    cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
    cell_metadata_ccf.set_index('cell_label', inplace=True)
    cell_labels = cell_metadata_ccf.index
    cell_labels = np.array(cell_labels)[::-1]

    # Views
    cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
    cell_metadata_file_views = os.path.join(DOWNLOAD_BASE, cell_metadata_path_views)
    cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
    cell_metadata_views.set_index('cell_label', inplace=True)

    # filtered_metadata_views = cell_metadata_views.loc[cell_labels]
    filtered_metadata_views = cell_metadata_views.loc[cell_labels]
    cells_class = filtered_metadata_views["class"].tolist()
    non_neuronal_mask = np.array(
        [True if any([j in i for j in NON_NEURONAL_CELL_TYPES]) else False for i in cells_class])

    # Fetch the transformed coordinates from the selected dataset
    transformed_coordinates = np.load(os.path.join(TRANSFORM_DIR,
                                                   f"all_transformed_cells_{ATLAS_USED}_{dataset_n}.npy"))
    transformed_coordinates_neurons = transformed_coordinates[~non_neuronal_mask]

    transformed_coordinates[:, [0, 1]] = transformed_coordinates[:, [1, 0]]  # Swap the 0 and 1 dims to match the atlas shape
    transformed_coordinates_neurons[:, [0, 1]] = transformed_coordinates_neurons[:, [1, 0]]  # Swap the 0 and 1 dims to match the atlas shape

    for n, reg in enumerate(unique_atlas_values):

        ut.print_c(f"[INFO] Fetching cells from reg {int(reg)}; {n+1}/{n_unique_atlas_values} regs; in dataset {dataset_n}!")
        reg_mask = CLIPPED_ATLAS == reg
        reg_mask_8bit = reg_mask.astype("uint8") * 255

        # All cells
        reg_cell_coordinates, _ = scm.filter_points_in_3d_mask(transformed_coordinates, reg_mask_8bit)

        n_cells_in_reg = len(reg_cell_coordinates)
        if reg in region_cells_counts:
            region_cells_counts[reg] += n_cells_in_reg
        else:
            region_cells_counts[reg] = n_cells_in_reg

        # Neurons
        reg_neuron_coordinates, _ = scm.filter_points_in_3d_mask(transformed_coordinates_neurons, reg_mask_8bit)
        n_neurons_in_reg = len(reg_neuron_coordinates)
        if reg in region_neuronal_counts:
            region_neuronal_counts[reg] += n_neurons_in_reg
        else:
            region_neuronal_counts[reg] = n_neurons_in_reg

        # Region voxel size
        if not reg in region_voxel_size:
            region_voxel_size[reg] = np.sum(reg_mask)

########################################################################################################################
# LOOP OVER THE SELECTED ATLAS PLANES
########################################################################################################################

step = 5
selected_planes = np.arange(85, 445+step, step)
clipped_atlas_coronal = np.swapaxes(CLIPPED_ATLAS, 0, 1)

# Get the viridis colormap
cmap = plt.cm.viridis

# Normalize the neuron counts to the range [0, 1] for colormap mapping
norm = plt.Normalize(vmin=0, vmax=0.78125)
# 0.78125 = 50 cells in 100um cube

for selected_plane in selected_planes:
    atlas_slice = clipped_atlas_coronal[selected_plane]
    atlas_slice_rgb = np.zeros((*atlas_slice.shape, 3), dtype=np.uint8)  # RGB image
    unique_ids = np.unique(atlas_slice)
    for reg in unique_ids:
        if reg != 0:
            reg_mask = atlas_slice == reg
            # Calculate the average neuron count in this region
            average_neuron_count = region_neuronal_counts[reg]/region_voxel_size[reg]
            # Map the average neuron count to the colormap
            color = cmap(norm(average_neuron_count))[:3]  # Get RGB values, ignore alpha
            # Assign the RGB color to the corresponding area in the RGB slice
            atlas_slice_rgb[reg_mask] = (np.array(color) * 255).astype(np.uint8)
    tifffile.imwrite(os.path.join(SAVING_DIRECTORY_SLICES, f"density_neurons_{selected_plane}_{ATLAS_USED}.tif"),
                     atlas_slice_rgb)
    tifffile.imwrite(os.path.join(SAVING_DIRECTORY_SLICES, f"atlas_16b_{selected_plane}_{ATLAS_USED}.tif"),
                     atlas_slice)

# Convert metric to 100um cubic
