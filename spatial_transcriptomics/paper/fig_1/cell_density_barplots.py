"""
This script generates barplots for cell densities from the combined ABC atlas datasets.

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
META_REGION_NAMES = ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", "TH", "HY", "MB", "P", "MY", "CB",
                     "fiber tracts", "VS"]

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
ATLAS_METADATA_FILE = os.path.join(ATLAS_DIR, f"{ATLAS_USED}_annotation_mouse.json")
ATLAS_METADATA = ut.load_json_file(ATLAS_METADATA_FILE)
REFERENCE_FILE = os.path.join(ATLAS_DIR, f"{ATLAS_USED}_reference_mouse.tif")
TRANSFORM_DIR = os.path.join(RESOURCES_DIR, "abc_atlas")

SAVING_DIRECTORY = ut.create_dir(fr"/default/path")  # PERSONAL

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
CLIPPED_ATLAS[:, :, int(ATLAS_SHAPE[-1] / 2):] = 0
CLIPPED_ATLAS_MASK = CLIPPED_ATLAS > 0
CLIPPED_REFERENCE = REFERENCE.copy()
CLIPPED_REFERENCE[:, :, int(ATLAS_SHAPE[-1] / 2):] = 0

########################################################################################################################
# LOOP OVER EVERY ATLAS ID AND GET THE CELL COUNT IN THE REGION MASK
########################################################################################################################
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
# LOOP OVER EVERY ATLAS ID AND GET THE CELL COUNT IN THE REGION MASK
########################################################################################################################

volume_voxel = 25 * 25 * 25  # in (um**3)
volume_100um_cube = 100 * 100 * 100  # in (um**3)

region_neuronal_densities = {
    reg: (region_neuronal_counts[reg] / region_voxel_size[reg]) / (volume_voxel / volume_100um_cube) for reg in
    region_neuronal_counts.keys()}
sorted_unique_atlas_values = sorted(region_neuronal_densities, key=region_neuronal_densities.get, reverse=True)
sorted_unique_atlas_values = [i for i in sorted_unique_atlas_values if i != 0]

sorted_reg_acronyms = [sut.find_dict_by_key_value(ATLAS_METADATA, reg).get("acronym")
                       if sut.find_dict_by_key_value(ATLAS_METADATA, reg)
                       else None for reg in sorted_unique_atlas_values]
sorted_reg_colors = [sut.find_dict_by_key_value(ATLAS_METADATA, reg).get("color_hex_triplet")
                     if sut.find_dict_by_key_value(ATLAS_METADATA, reg)
                     else None for reg in sorted_unique_atlas_values]
sorted_reg_colors = np.array(['#' + color for color in sorted_reg_colors])
sorted_reg_density = [region_neuronal_densities[reg] for reg in sorted_unique_atlas_values]

# Exclude the key 0
filtered_cells_counts = {k: v for k, v in region_neuronal_counts.items() if k != 0}
filtered_voxel_size = {k: v for k, v in region_voxel_size.items() if k != 0}
avg_neuronal_density = sum(filtered_cells_counts.values()) / sum(filtered_voxel_size.values())
avg_cells_in_100um_cube = avg_neuronal_density / (
        volume_voxel / volume_100um_cube)  # cells in a cube of side length 100um

# Plot params
x_tick_fs = 6

# Create a bar plot
plt.figure(figsize=(15, 3))  # Adjust the figure size as needed
bars = plt.bar(np.array(sorted_reg_acronyms),
               np.array(sorted_reg_density),
               color=np.array(sorted_reg_colors), linewidth=0.1, edgecolor='black',
               width=1)
# Add a horizontal line at the average value
plt.axhline(y=avg_cells_in_100um_cube, color='red', linestyle='--', linewidth=1.5,
            label=f'Average: {avg_cells_in_100um_cube:.2f}')
# plt.title('Cell Counts in Brain Regions')
# plt.xlabel('Brain Region')
# plt.ylabel(f'{saving_name} neurons')
plt.xticks(rotation=45, ha="right", fontsize=x_tick_fs)
plt.tight_layout()  # Adjust layout to not cut off labels
plt.savefig(os.path.join(SAVING_DIRECTORY, f"density_neurons_{ATLAS_USED}.png"), dpi=300)
plt.savefig(os.path.join(SAVING_DIRECTORY, f"density_neurons_{ATLAS_USED}.svg"), dpi=300)
# plt.show()
plt.close()

########################################################################################################################
# LOOP OVER EVERY METAREGION AND GET THE CELL COUNT IN THE REGION MASK
########################################################################################################################

meta_region_ids = []
meta_region_neuron_densities = []

for meta_region in META_REGION_NAMES:
    meta_region_counts = []
    meta_region_voxels = []
    meta_region = sut.find_dict_by_key_value(ATLAS_METADATA, meta_region, key="acronym")
    meta_region_id = meta_region["id"]
    meta_region_ids.append(meta_region_id)
    meta_region_children = sut.find_child_ids(ATLAS_METADATA, meta_region_id)
    if meta_region_id in region_neuronal_counts:
        meta_region_count = region_neuronal_counts[meta_region_id]
        meta_region_counts.append(meta_region_count)
        meta_region_voxel = region_voxel_size[meta_region_id]
        meta_region_voxels.append(meta_region_voxel)
    for child_id in meta_region_children:
        if child_id in region_neuronal_counts:
            child_counts = region_neuronal_counts[child_id]
            meta_region_counts.append(child_counts)
            child_voxels = region_voxel_size[child_id]
            meta_region_voxels.append(child_voxels)
    meta_region_neuron_density = (np.sum(meta_region_counts) / np.sum(meta_region_voxels)) / (
                volume_voxel / volume_100um_cube)
    meta_region_neuron_densities.append(meta_region_neuron_density)

# Neurons
meta_region_acronyms = [sut.find_dict_by_key_value(ATLAS_METADATA, i).get("acronym")
                        if sut.find_dict_by_key_value(ATLAS_METADATA, i)
                        else None for i in meta_region_ids]
meta_region_colors = [sut.find_dict_by_key_value(ATLAS_METADATA, i).get("color_hex_triplet")
                      if sut.find_dict_by_key_value(ATLAS_METADATA, i)
                      else None for i in meta_region_ids]
meta_region_colors = np.array(['#' + color for color in meta_region_colors])

# Create a bar plot
plt.figure(figsize=(7, 3))  # Adjust the figure size as needed
bars = plt.bar(np.array(meta_region_acronyms),
               np.array(meta_region_neuron_densities),
               color=meta_region_colors, linewidth=0.5, edgecolor='black',
               width=1)
plt.ylabel(f'density cells')
plt.xticks(rotation=45, ha="right", fontsize=x_tick_fs)
plt.tight_layout()  # Adjust layout to not cut off labels
plt.savefig(os.path.join(SAVING_DIRECTORY, f"density_neurons_meta_regions_{ATLAS_USED}.png"), dpi=300)
plt.savefig(os.path.join(SAVING_DIRECTORY, f"density_neurons_meta_regions_{ATLAS_USED}.svg"), dpi=300)
# plt.show()
