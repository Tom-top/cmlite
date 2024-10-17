import os

import numpy as np
import pandas as pd
import tifffile
import nibabel as nib

import utils.utils as ut

import IO.file_utils as fu
import resampling.resampling as res


def hex_to_rgb(hex_color):
    """Convert hex color (e.g., '#ff0000') to an RGB tuple (255, 0, 0)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


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


def check_within_mask(mask, raw_cells):
    valid_cells = []
    invalid_cells = []
    n_cells = len(raw_cells)
    for n, cell in enumerate(raw_cells):
        ut.print_c(f"[INFO] Filtering cell: {n+1}/{n_cells}")
        z, y, x = cell
        # Ensure the coordinates are within the bounds of the mask and check if the mask value is greater than 0
        if (0 <= x < mask.shape[0] and
                0 <= y < mask.shape[1] and
                0 <= z < mask.shape[2] and
                mask[int(x), int(y), int(z)] > 0):
            valid_cells.append(cell)
        else:
            invalid_cells.append(cell)

    return np.array(valid_cells)

########################################################################################################################
# SETTING ESSENTIAL DIRECTORIES
########################################################################################################################

working_directory = r"E:\tto\c-Fos_color_labels\ID013_an000405_g002_Brain_M4_rescan1"
raw_data_path = os.path.join(working_directory, "raw_data.tif")
raw_data_clipped_path = os.path.join(working_directory, "raw_data_clipped.tif")
raw_image_resolution = (5, 5, 10)  # The resolution of the analyzed data

########################################################################################################################
# CROP THE IMAGE IF A BOUNDING BOX WAS DEFINED
########################################################################################################################

bounding_box_path = os.path.join(working_directory, "bbox.npz")
if os.path.exists(bounding_box_path) and not os.path.exists(raw_data_clipped_path):
    ut.print_c(f"[INFO] Bounding box file detected! Applying crop")
    raw_data = tifffile.imread(raw_data_path)
    bounding_box = np.load(bounding_box_path, allow_pickle=True)
    bounding_box = bounding_box["bbox"]
    cropped_data = raw_data[bounding_box[0], bounding_box[1], bounding_box[2]]
    tifffile.imwrite(raw_data_clipped_path, cropped_data)
elif os.path.exists(raw_data_clipped_path):
    ut.print_c(f"[INFO] Cropped data already exists, loading it!")
    cropped_data = tifffile.imread(raw_data_clipped_path)
cropped_data = np.swapaxes(cropped_data, 0, 2)

########################################################################################################################
# UPSAMPLE THE TISSUE MASK
########################################################################################################################

upsampled_tissue_mask_path = os.path.join(working_directory, "upsampled_tissue_mask.tif")

if not os.path.exists(upsampled_tissue_mask_path):
    ut.print_c("[INFO] Upsampling the tissue mask!")
    tissue_mask_path = os.path.join(working_directory, "atlas_mask_tissue_full.nii.gz")
    tissue_mask = nib.load(tissue_mask_path).get_fdata().astype("uint8")
    upsample_tissue_mask_parameter = {
        "source_resolution": (25, 25, 25),
        "sink_resolution": raw_image_resolution,
        "sink_shape": cropped_data.shape,
        "processes": None,
        "verbose": True,
        "interpolation": "nearest",
        "method": "memmap",
    }
    fu.delete_file(upsampled_tissue_mask_path)
    res.resample(tissue_mask, sink=upsampled_tissue_mask_path,
                 **upsample_tissue_mask_parameter)
else:
    ut.print_c("[INFO] Upsampled tissue mask already exists!")

########################################################################################################################
#  FETCH THE RAW COORDINATES
########################################################################################################################

raw_cell_path = os.path.join(working_directory, "raw_cells.npy")
if not os.path.exists(raw_cell_path):
    signal_raw_path = os.path.join(working_directory, "signal_642_raw.nii.gz")
    signal_raw = nib.load(signal_raw_path).get_fdata()
    raw_cells = np.where(signal_raw > 0)
    np.save(raw_cell_path, raw_cells)
else:
    raw_cells = np.load(raw_cell_path)

# Transpose to get the coordinates in rows (X, Y, Z)
coordinates = np.array(raw_cells).T

# Filter the cell coordinates that are valid based on tissue mask
mask = tifffile.imread(upsampled_tissue_mask_path)
coordinates = check_within_mask(mask, coordinates)

# Change coordinates from pixel --> physical distance
coordinates[:, 0] = coordinates[:, 0] * raw_image_resolution[0]
coordinates[:, 1] = coordinates[:, 1] * raw_image_resolution[1]
coordinates[:, 2] = coordinates[:, 2] * raw_image_resolution[2]

# Create a DataFrame with ID, T (set to 0 for each point), X, Y, and Z columns
df = pd.DataFrame(coordinates, columns=['X', 'Y', 'Z'])
df['ID'] = np.arange(1, len(df) + 1)  # Create unique IDs for each point
df['T'] = 0  # Set time to 0 for all points
df = df[['ID', 'T', 'X', 'Y', 'Z']]

# Define the output CSV path
csv_output_path = os.path.join(working_directory, "raw_cells_tracks_aivia.csv")
# Save the DataFrame to CSV format
df.to_csv(csv_output_path, index=False)
