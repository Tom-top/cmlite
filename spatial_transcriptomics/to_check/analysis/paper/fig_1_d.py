import os

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import requests
import json

from ClearMap.Environment import *


def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Attempt to load the JSON content
            return json.load(file)
    except FileNotFoundError:
        print("Error: The file was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def find_dict_by_id_value(data, target_id, key="id"):
    # If data is a dictionary, check if it contains the key with the matching target value
    if isinstance(data, dict):
        if key in data and data[key] == target_id:
            # Return the entire dictionary if key with matching value is found
            return data
        else:
            # Otherwise, search recursively in values
            for v in data.values():
                result = find_dict_by_id_value(v, target_id, key=key)
                if result is not None:
                    return result

    # If data is a list, iterate and search recursively in each item
    elif isinstance(data, list):
        for item in data:
            result = find_dict_by_id_value(item, target_id, key=key)
            if result is not None:
                return result

    # Return None if the key with the matching value is not found in any dictionary
    return None


def sort_cells_and_get_indices(cell_counts):
    # Convert the list to a numpy array for easier manipulation
    cell_counts_array = np.array(cell_counts)
    # Get indices that would sort the array from lowest to highest
    ascending_indices = np.argsort(cell_counts_array)
    # Reverse the indices to get highest to lowest
    descending_indices = ascending_indices[::-1]
    # Sort the cell counts array using the descending indices
    sorted_cell_counts = cell_counts_array[descending_indices]

    return sorted_cell_counts, descending_indices


def hex_to_rgb(hex_value, alpha=False):
    # Strip the leading '#' if it exists
    hex_value = hex_value.lstrip('#')

    # Convert the three parts from hex to integers
    r, g, b = int(hex_value[0:2], 16), int(hex_value[2:4], 16), int(hex_value[4:6], 16)

    if alpha:
        return (r, g, b, 255)
    else:
        return (r, g, b)


working_dir = "/mnt/data/spatial_transcriptomics/results"

dataset = "mpd5_pick1"
dataset_dir = os.path.join(working_dir, dataset)
comparison = "Pick-1_vs_Vehicle"
comparison_dir = os.path.join(dataset_dir, comparison)
region_dir = os.path.join(comparison_dir, "region_figures")

bin_data_path = os.path.join(comparison_dir, "bin_whole.tif")
bin_data = tifffile.imread(bin_data_path)

ressources_path = "/home/imaging/PycharmProjects/ClearMap2/ClearMap/Resources/Atlas"
atlas_path = os.path.join(ressources_path, "atlas_annotations_coronal.tif")
atlas = tifffile.imread(atlas_path)
masked_atlas = atlas[bin_data]
unique_masked_atlas_values = np.unique(masked_atlas)
total_unique_masked_atlas_values = [np.sum(atlas == i) for i in unique_masked_atlas_values]
total_unique_masked_atlas_values = np.array(total_unique_masked_atlas_values)

atlas_metadata_path = os.path.join(ressources_path, "Gubra_annotation.json")
atlas_metadata_gubra = load_json_file(atlas_metadata_path)

atlas_acro = [find_dict_by_id_value(atlas_metadata_gubra, i).get("acronym")
              if find_dict_by_id_value(atlas_metadata_gubra, i)
              else None for i in unique_masked_atlas_values]

# Check for None
# all_ids = np.arange(5000, 6328 + 1, 1)
# atlas_acro_whole_atlas = [find_dict_by_id_value(atlas_metadata_gubra, i).get("acronym")
#                           if find_dict_by_id_value(atlas_metadata_gubra, i)
#                           else None for i in all_ids]
# for i, j in zip(atlas_acro_whole_atlas, all_ids):
#     if i is None:
#         print(f"Check id {j}: missing data in json file!")

atlas_acro = np.array(atlas_acro)
atlas_hex = [find_dict_by_id_value(atlas_metadata_gubra, i).get("color_hex_triplet")
             if find_dict_by_id_value(atlas_metadata_gubra, i)
             else None for i in unique_masked_atlas_values]
atlas_hex = np.array(atlas_hex)

n_masked_atlas_values = []
for i in unique_masked_atlas_values:
    c = np.sum(masked_atlas == i)
    n_masked_atlas_values.append(c)
n_masked_atlas_values = np.array(n_masked_atlas_values)

none_mask = np.array(atlas_acro) == None
n_masked_atlas_values = n_masked_atlas_values[~none_mask]
sorted_n_masked_atlas_values, sorted_n_masked_atlas_idx = sort_cells_and_get_indices(n_masked_atlas_values)
sorted_unique_atlas_acro = atlas_acro[~none_mask][sorted_n_masked_atlas_idx]
sorted_unique_atlas_hex = atlas_hex[~none_mask][sorted_n_masked_atlas_idx]
sorted_unique_atlas_hex = np.array(['#' + color for color in sorted_unique_atlas_hex])
total_unique_masked_atlas_values = total_unique_masked_atlas_values[~none_mask][sorted_n_masked_atlas_idx]

# Create a bar plot
n_bars = 50
tick_fs = 6
plt.figure(figsize=(8, 5))  # Adjust the figure size as needed
bars = plt.bar(sorted_unique_atlas_acro[:n_bars],
               sorted_n_masked_atlas_values[:n_bars],
               color=sorted_unique_atlas_hex[:n_bars], linewidth=0.2, edgecolor='black',
               width=1)
# plt.title('Cell Counts in Brain Regions')
# plt.xlabel('Brain Region')
plt.xlim(-0.5, n_bars - 0.5)
plt.xticks(rotation=45, ha="right", fontsize=tick_fs)
plt.yticks(fontsize=tick_fs)
plt.ylabel("Differentially active voxels in region")
plt.savefig(os.path.join(comparison_dir, f"active_voxels_{n_bars}.png"), dpi=300)
plt.savefig(os.path.join(comparison_dir, f"active_voxels_{n_bars}.svg"), dpi=300)
# plt.show()

#
n_bars = 100
tick_fs = 6
plt.figure(figsize=(8, 5))  # Adjust the figure size as needed
proportion_masked_atlas_values = sorted_n_masked_atlas_values / total_unique_masked_atlas_values
sorted_proportion_values, sorted_proportion_idx = sort_cells_and_get_indices(proportion_masked_atlas_values)
bars = plt.bar(sorted_unique_atlas_acro[sorted_proportion_idx][:n_bars],
               sorted_proportion_values[:n_bars] * 100,
               color=sorted_unique_atlas_hex[sorted_proportion_idx][:n_bars], linewidth=0.2, edgecolor='black',
               width=1)
# plt.title('Cell Counts in Brain Regions')
# plt.xlabel('Brain Region')
plt.xlim(-0.5, n_bars - 0.5)
plt.xticks(rotation=45, ha="right", fontsize=tick_fs)
plt.yticks(fontsize=tick_fs)
plt.ylabel("Differentially active voxels in region (%)")
plt.savefig(os.path.join(comparison_dir, f"percent_active_voxels_{n_bars}.png"), dpi=300)
plt.savefig(os.path.join(comparison_dir, f"percent_active_voxels_{n_bars}.svg"), dpi=300)
# plt.show()

# Create an RGB version of the binary mask
valid_unique_masked_atlas_values = unique_masked_atlas_values[~none_mask][sorted_n_masked_atlas_idx]
atlas_rgb = np.full_like(atlas, 0)
atlas_rgb = np.repeat(atlas_rgb[..., np.newaxis], 3, axis=-1).astype("uint8")

for v, c in zip(valid_unique_masked_atlas_values, sorted_unique_atlas_hex):
    atlas_mask = atlas == v
    atlas_rgb[np.logical_and(atlas_mask, bin_data)] = hex_to_rgb(c)

tifffile.imwrite(os.path.join(comparison_dir, "bin_whole_rgb.tif"), atlas_rgb)

# Extract mask and colored bin version of each region

reference_file = os.path.join(ressources_path, "atlas_template_coronal.tif")
reference = tifffile.imread(reference_file) / 7
inverted_ref = 255 - reference
reference_rgb = np.repeat(inverted_ref[..., np.newaxis], 3, axis=-1).astype("uint8")

sorted_unique_masked_atlas_values = unique_masked_atlas_values[~none_mask][sorted_n_masked_atlas_idx][
    sorted_proportion_idx]
region_rgb = np.full_like(atlas, 255)
region_rgb = np.repeat(region_rgb[..., np.newaxis], 3, axis=-1).astype("uint8")

for id, hex_c, acro in zip(sorted_unique_masked_atlas_values[:n_bars],
                           sorted_unique_atlas_hex[sorted_proportion_idx][:n_bars],
                           sorted_unique_atlas_acro[sorted_proportion_idx][:n_bars]):

    print(f"Generating figures for: {acro}")

    reg_dir = os.path.join(region_dir, acro)
    if not os.path.exists(reg_dir):
        os.mkdir(reg_dir)

    region_mask = atlas == id
    region_rgb_copy = region_rgb.copy()
    region_rgb_copy[region_mask] = np.array([0, 0, 0])
    region_rgb_copy[np.logical_and(region_mask, bin_data)] = hex_to_rgb(hex_c)
    if os.path.exists(os.path.join(reg_dir, f"{acro}_reg_and_mask.tif")):
        print("Duplicated area detected!")
        tifffile.imwrite(os.path.join(reg_dir, f"{acro}_reg_and_mask_2.tif"), region_rgb_copy)
    else:
        tifffile.imwrite(os.path.join(reg_dir, f"{acro}_reg_and_mask.tif"), region_rgb_copy)

    from scipy.ndimage import binary_erosion

    for ori, ori_n in zip(np.arange(0, 3, 1), ["coronal", "sagittal", "horizontal"]):
        template_max_proj = np.min(reference_rgb, axis=ori)

        region_coronal_bin = region_rgb.copy()
        region_coronal_bin[region_mask] = np.array([0, 0, 0])
        region_coronal_bin_max_proj = np.min(region_coronal_bin, axis=ori)
        mask = (region_coronal_bin_max_proj == [0, 0, 0]).all(axis=2)
        eroded_mask = binary_erosion(mask)
        outline = mask ^ eroded_mask
        # tifffile.imwrite(os.path.join(comparison_dir, f"BST_outline_{ori_n}.tif"), outline)

        region_coronal_rgb = region_rgb.copy()
        region_coronal_rgb[np.logical_and(region_mask, bin_data)] = hex_to_rgb(hex_c)
        # tifffile.imwrite(os.path.join(reg_dir, f"test.tif"), region_rgb_copy)
        region_coronal_bin_max_proj = np.min(region_coronal_rgb, axis=ori)
        # tifffile.imwrite(os.path.join(reg_dir, f"test2.tif"), region_coronal_bin_max_proj)
        mask_rgb = (region_coronal_bin_max_proj < [255, 255, 255]).any(axis=2)
        template_max_proj[mask_rgb] = hex_to_rgb(hex_c)
        # tifffile.imwrite(os.path.join(reg_dir, f"test3.tif"), template_max_proj)
        template_max_proj[outline] = np.array([0, 0, 0])
        # region_coronal_bin_max_proj[outline] = np.array([0, 0, 0])
        # tifffile.imwrite(os.path.join(comparison_dir, f"BST_outline_and_mask_{ori_n}.tif"), region_coronal_bin_max_proj)
        if os.path.exists(os.path.join(reg_dir, f"{acro}_outline_and_mask_{ori_n}.tif")):
            tifffile.imwrite(os.path.join(reg_dir, f"{acro}_outline_and_mask_{ori_n}_2.tif"), template_max_proj)
        else:
            tifffile.imwrite(os.path.join(reg_dir, f"{acro}_outline_and_mask_{ori_n}.tif"), template_max_proj)
        # break
    # break
