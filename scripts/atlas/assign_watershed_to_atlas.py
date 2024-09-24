import os
import json
import numpy as np
import tifffile
import utils.utils as ut

def hex_to_rgb(hex_color):
    """Convert hex color (e.g., '#ff0000') to an RGB tuple (255, 0, 0)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_label_color_map(metadata, unique_labels):
    """Precompute a label-to-color map for quick lookup."""
    label_color_map = {}
    for label in unique_labels:
        if label == 0 or label == 5000:  # Skip background or unlabeled regions
            continue
        region_info = find_region_by_id(metadata, label)
        if region_info:
            label_color_map[label] = hex_to_rgb(f'#{region_info["color_hex_triplet"]}')
        else:
            print(f"[WARNING] skipping region with label: {label}")
    return label_color_map

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

# File paths
working_directory = r"E:\tto\23-GUP030-0696\raw\ID888_an000888_g010_brain_M3\xy5p0_z5p0\2024-08-29_194534_merged\figures"
watershed_path = os.path.join(working_directory, "shape_detection_filtered.tif")
atlas_to_auto_path = r"E:\tto\23-GUP030-0696\raw\ID888_an000888_g010_brain_M3\xy5p0_z5p0\2024-08-29_194534_merged\mouse_gubra_atlas_to_auto_3\result_5um.tif"
atlas_path = r"resources\atlas"

# Load data
watershed = tifffile.imread(watershed_path)
atlas_to_auto_array = np.swapaxes(tifffile.imread(atlas_to_auto_path), 0, 2)
annotation = tifffile.imread(os.path.join(atlas_path, "gubra_annotation_mouse.tif"))

# Create an RGB image to store the colored watershed, initialized with zeros
colored_watershed = np.zeros((*watershed.shape, 3), dtype=np.uint8)

# Load metadata
with open(os.path.join(atlas_path, "gubra_annotation_mouse.json"), "r") as f:
    metadata = json.load(f)["msg"][0]

# Precompute the color map for unique labels
unique_labels = sorted(np.unique(annotation))
label_color_map = create_label_color_map(metadata, unique_labels)
print(f"[INFO] {len(unique_labels)} unique labels detected")

# Process each label and color the watershed accordingly
for label, color in label_color_map.items():
    ut.print_c(f"[INFO] Labelling cells in region {label}")

    # Create a mask for this region and watershed in a vectorized manner
    region_mask = (atlas_to_auto_array == label)
    watershed_in_mask = np.logical_and((watershed > 0), region_mask)

    # Apply the region color to the corresponding pixels in the colored watershed image
    colored_watershed[watershed_in_mask] = color

# Save the colored watershed image
output_path = os.path.join(working_directory, "colored_watershed_optimized.tif")
tifffile.imwrite(output_path, colored_watershed)

print(f"[INFO] Optimized colored watershed saved to {output_path}")
