import os
import json

import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("Agg")
matplotlib.use("Qt5Agg")


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

transformed_cells_data_dir = r"E:\tto\23-GUP030-0696\raw\ID888_an000888_g010_brain_M3\xy5p0_z5p0" \
                             r"\2024-08-29_194534_merged\shape_detection_350"
transformed_cells_data_path = os.path.join(transformed_cells_data_dir, "mouse_gubra_cells_transformed_3.csv")
transformed_cells_data = pd.read_csv(transformed_cells_data_path, delimiter=";", quotechar='"')

annotation = tifffile.imread(os.path.join(atlas_path, "gubra_annotation_mouse.tif"))
unique_labels = sorted(np.unique(annotation))
print(f"[INFO] {len(unique_labels)} unique labels detected")

# Initialize dictionaries to hold counts for left and right sides
counts = {}
density_counts = {}
colors = {}
volume = {}
intensity = {}
ordered_acronyms = []

# Count pixels for each label
for label in unique_labels:

    if label == 0:
        continue  # Assuming 0 is the background or unlabeled region
    region_info = find_region_by_id(metadata, label)

    if region_info is None:
        print(f"[WARNING] skipping region with label: {label}")
    else:
        region_volume = np.sum(annotation == label)
        region_name = region_info["name"]
        region_acronym = region_info["acronym"]
        region_color = f'#{region_info["color_hex_triplet"]}'

        ordered_acronyms.append(region_acronym)
        mask_cells_in_region = transformed_cells_data["name"] == region_name
        transformed_cells_in_region_sum = np.sum(mask_cells_in_region)
        volume[region_acronym] = np.mean(transformed_cells_data["size"][mask_cells_in_region])
        intensity[region_acronym] = np.mean(transformed_cells_data["source"][mask_cells_in_region])
        counts[region_acronym] = transformed_cells_in_region_sum
        density_counts[region_acronym] = transformed_cells_in_region_sum / region_volume  # Cells per voxel
        colors[region_acronym] = region_color

# Prepare data for plotting
regions = [region for region in ordered_acronyms if region in counts]
count_values = [counts[region] for region in regions]
density_counts = [density_counts[region] for region in regions]
volume = [volume[region] for region in regions]
intensity = [intensity[region] for region in regions]
bar_colors = [colors[region] for region in regions]

regions = np.array(regions)
count_values = np.array(count_values)
density_counts = np.array(density_counts)
volume = np.array(volume)
intensity = np.array(intensity)
bar_colors = np.array(bar_colors)

sorted_indices = np.argsort(density_counts)
sorted_regions = regions[sorted_indices]
sorted_count_values = count_values[sorted_indices]
sorted_density_values = density_counts[sorted_indices]
sorted_bar_colors = bar_colors[sorted_indices]

mean_density = np.mean(density_counts)

########################################################################################################################
# Bar plot: density all brain regions rostro-caudal
########################################################################################################################

# Plot the data
fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(25, 3))

# Plot regions
ax.bar(regions[::-1], density_counts[::-1], color=bar_colors[::-1], width=0.8)
# ax.set_xlim(0, 80)
ax.set_xlabel('Microglia density per region')
ax.invert_xaxis()  # Flip the x-axis to make the bars grow leftwards

# Add a black horizontal line representing the mean of density_values
ax.axhline(mean_density, color='black', linestyle='--', linewidth=1)

# Align the y-axis labels
fig.tight_layout()
plt.savefig(os.path.join(transformed_cells_data_dir,
                         "microglial_density_per_region.png"), dpi=300)
plt.savefig(os.path.join(transformed_cells_data_dir,
                         "microglial_density_per_region.svg"), dpi=300)
plt.show()

########################################################################################################################
# Bar plot: density all brain sorted
########################################################################################################################

# Plot the data
fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(25, 3))

# Plot regions
ax.bar(sorted_regions, sorted_density_values, color=sorted_bar_colors, width=0.8)
ax.set_xlabel('Microglia density per region')
ax.invert_xaxis()  # Flip the x-axis to make the bars grow leftwards

# Add a black horizontal line representing the mean of density_values

ax.axhline(mean_density, color='black', linestyle='--', linewidth=1)

# Align the y-axis labels
fig.tight_layout()
plt.savefig(os.path.join(transformed_cells_data_dir,
                         "microglial_density_per_region_sorted.png"), dpi=300)
plt.savefig(os.path.join(transformed_cells_data_dir,
                         "microglial_density_per_region_sorted.svg"), dpi=300)
plt.show()

########################################################################################################################
# Bar plot: mean volume of cells across all brain regions
########################################################################################################################

# Plot the data
fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(25, 3))

# Plot regions
ax.bar(regions[::-1], volume[::-1], color=bar_colors[::-1], width=0.8)
ax.set_xlabel('Microglia volume per region (average volume of cells)')
ax.invert_xaxis()  # Flip the x-axis to make the bars grow leftwards

# Align the y-axis labels
fig.tight_layout()
plt.savefig(os.path.join(transformed_cells_data_dir,
                         "microglial_volume_per_region.png"), dpi=300)
plt.savefig(os.path.join(transformed_cells_data_dir,
                         "microglial_volume_per_region.svg"), dpi=300)
plt.show()

########################################################################################################################
# Bar plot: mean intensity of cells across all brain regions
########################################################################################################################

# Plot the data
fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(25, 3))

# Plot regions
ax.bar(regions[::-1], intensity[::-1], color=bar_colors[::-1], width=0.8)
ax.set_xlabel('Microglia intensity per region (average volume of cells)')
ax.invert_xaxis()  # Flip the x-axis to make the bars grow leftwards

# Align the y-axis labels
fig.tight_layout()
plt.savefig(os.path.join(transformed_cells_data_dir,
                         "microglial_intensity_per_region.png"), dpi=300)
plt.savefig(os.path.join(transformed_cells_data_dir,
                         "microglial_intensity_per_region.svg"), dpi=300)
plt.show()

########################################################################################################################
# Bar plot: density all brain sorted
########################################################################################################################

volume = transformed_cells_data["size"]
intensity = transformed_cells_data["source"]

# Create a figure with multiple subplots
fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(25, 3))

# Histogram
x_max = 100
ax.hist(volume, bins=x_max, range=(0, x_max), width=1, color='#FF6961', edgecolor='black')
ax.set_xlim(0, x_max)
ax.set_title('Histogram of Cell Volumes')
ax.set_xlabel('Volume')
ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(transformed_cells_data_dir,
                         "microglial_volume_distribution.png"), dpi=300)
plt.savefig(os.path.join(transformed_cells_data_dir,
                         "microglial_volume_distribution.svg"), dpi=300)
plt.show()
