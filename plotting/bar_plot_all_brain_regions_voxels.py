import os
import json

import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
# matplotlib.use("Qt5Agg")

import utils.utils as ut


def find_region_by_id(region_data, target_id):
    if isinstance(region_data, dict):
        if region_data.get('id') == target_id:
            return region_data
        for child in region_data.get('children', []):
            result = find_region_by_id(child, target_id)
            if result:
                return result
    elif isinstance(region_data, list):
        for item in region_data:
            result = find_region_by_id(item, target_id)
            if result:
                return result
    return None


atlas_path = r"resources\atlas"

with open(os.path.join(atlas_path, "gubra_annotation_mouse.json"), "r") as f:
    metadata = json.load(f)
    metadata = metadata["msg"][0]

map_name = "semaglutide_zscore_glp1r_merfish_overlap"
map_color = "magenta"
working_directory = r"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results\gene_expression\Semaglutide_Glp1r"
saving_directory = ut.create_dir(os.path.join(working_directory, "voxel_wise_intensity_distribution"))
transformed_heatmap_path = os.path.join(working_directory, "semaglutide_zscore_glp1r_merfish_overlap_sagittal.tif")
transformed_heatmap = tifffile.imread(transformed_heatmap_path)

annotation = tifffile.imread(os.path.join(atlas_path, "gubra_annotation_mouse.tif"))
unique_labels = sorted(np.unique(annotation))
print(f"[INFO] {len(unique_labels)} unique labels detected")

# Initialize dictionaries to hold counts for left and right sides
mean_voxel_intensity = {}
colors = {}
ordered_acronyms = []
hemisphere = True

# Count pixels for each label
for label in unique_labels:

    if label == 0:
        continue  # Assuming 0 is the background or unlabeled region
    region_info = find_region_by_id(metadata, label)

    if region_info is None:
        print(f"[WARNING] skipping region with label: {label}")
    else:
        region_mask = annotation == label
        region_volume = np.sum(region_mask)
        region_name = region_info["name"]
        region_acronym = region_info["acronym"]
        region_color = f'#{region_info["color_hex_triplet"]}'

        ordered_acronyms.append(region_acronym)
        mask_voxels_in_region = transformed_heatmap[region_mask]
        if hemisphere:
            transformed_voxels_in_region_mean = np.mean(mask_voxels_in_region) * 2
        else:
            transformed_voxels_in_region_mean = np.mean(mask_voxels_in_region)
        mean_voxel_intensity[region_acronym] = transformed_voxels_in_region_mean
        colors[region_acronym] = region_color

# Prepare data for plotting
regions = [region for region in ordered_acronyms if region in mean_voxel_intensity]
count_values = [mean_voxel_intensity[region] for region in regions]
bar_colors = [colors[region] for region in regions]

regions = np.array(regions)
count_values = np.array(count_values)
bar_colors = np.array(bar_colors)

sorted_indices = np.argsort(count_values)
sorted_regions = regions[sorted_indices]
sorted_count_values = count_values[sorted_indices]
sorted_bar_colors = bar_colors[sorted_indices]

mean_counts = np.mean(count_values)

########################################################################################################################
# Bar plot: density all brain regions rostro-caudal
########################################################################################################################

if map_color:
    color = map_color
else:
    color = bar_colors[::-1]

# Find the indices of the top 20 values
sorted_indices = np.argsort(count_values)[::-1][:20]

# Plot the data
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 5))

# Plot all regions with their corresponding counts and colors
bars = ax.bar(regions[::-1], count_values[::-1], color=color, width=0.8)

# Add a black horizontal line representing the mean of count_values
ax.axhline(mean_counts, color='black', linestyle='--', linewidth=1)

# Flip the x-axis to make the bars grow leftwards
ax.invert_xaxis()

# Set labels and layout
ax.set_xlabel('Intensity per region')
ax.set_ylabel('Mean Voxel Intensity')

# Only label the top 20 regions with their names
for idx in sorted_indices:
    bar = bars[::-1][idx]  # Reverse the order to match the plot
    ax.text(bar.get_x() + bar.get_width(), bar.get_height(), f'{regions[idx]}',
            ha='left', va='center', fontsize=10)

# Adjust the layout
fig.tight_layout()

# Save the plot
plt.savefig(os.path.join(saving_directory, f"{map_name}_per_region_top20_with_labels.png"), dpi=300)
plt.savefig(os.path.join(saving_directory, f"{map_name}_per_region_top20_with_labels.svg"), dpi=300)

plt.show()

# ########################################################################################################################
# # Bar plot: density all brain sorted
# ########################################################################################################################
#
# if map_color:
#     color = map_color
# else:
#     color = sorted_bar_colors
#
# # Plot the data
# fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(25, 3))
#
# # Plot regions
# ax.bar(sorted_regions, sorted_count_values, color=sorted_bar_colors, width=0.8)
# ax.set_xlabel('Intensity per region')
# ax.invert_xaxis()  # Flip the x-axis to make the bars grow leftwards
#
# # Add a black horizontal line representing the mean of density_values
#
# ax.axhline(mean_counts, color='black', linestyle='--', linewidth=1)
#
# # Set the y-range
# # ax.set_ylim(0, 0.025)
# # ax.set_ylim(0, 0.1)
#
# # Align the y-axis labels
# # fig.tight_layout()
# plt.savefig(os.path.join(saving_directory,
#                          f"{map_name}_per_region_sorted.png"), dpi=300)
# plt.savefig(os.path.join(saving_directory,
#                          f"{map_name}_per_region_sorted.svg"), dpi=300)
# plt.show()
