import os

import numpy as np
import tifffile
import nibabel as nib

import utils.utils as ut

# Define paths and cutoff value for binarization
zscore_maps_directory = r"resources\z-score_maps"  # PERSONAL
zscore_map_name = "Semaglutide"
zscore_map_directory = os.path.join(zscore_maps_directory, zscore_map_name)
zscore_map_path = os.path.join(zscore_map_directory, "result.nii.gz")
analysis_directory = ut.create_dir(fr"E:\tto\spatial_transcriptomics_results\{zscore_map_name}")  # PERSONAL
cutoff = 350  # Value above which all pixels will be kept for the mask (z-score * 100)

# Load and transpose the image
zscore_map_nii = nib.load(zscore_map_path)
zscore_map = zscore_map_nii.get_fdata()
zscore_map = np.flip(np.transpose(zscore_map, (1, 2, 0)), 1)

# Binarize the image based on the cutoff
zscore_map_bin = (zscore_map >= cutoff).astype("uint8")

# Process the first half of the image
midline = zscore_map.shape[-1] // 2
zscore_map_bin_half = zscore_map_bin[:, :, :midline]

# Set the binary values to 255
zscore_map_bin[zscore_map_bin == 1] = 255
zscore_map_bin_half[zscore_map_bin_half == 1] = 255

# Save the binarized images
tifffile.imwrite(os.path.join(analysis_directory, "hemisphere_mask.tif"), zscore_map_bin_half)
tifffile.imwrite(os.path.join(analysis_directory, "whole_brain_mask.tif"), zscore_map_bin)
