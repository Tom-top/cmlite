import os
import numpy as np
import tifffile

import utils.utils as ut

# Define paths and regions to binarize
ref_directory = r"resources\atlas"
# ref_directory = r"E:\tto\test"
ref_path = os.path.join(ref_directory, "gubra_annotation_mouse.tif")
# ref_path = os.path.join(ref_directory, "ano.tif")
regions_to_bin = [5825, 5824, 5868]  # List of region ids to include in the binary mask
brain_area_name = "VTA"
analysis_directory = ut.create_dir(fr"/default/path")  # PERSONAL

# Load and transpose the image
annotation = np.transpose(tifffile.imread(ref_path), (1, 2, 0))

# Binarize and combine masks
combined_regions_bin = np.logical_or.reduce([(annotation == region).astype(np.uint8) for region in regions_to_bin])\
    .astype("uint8")

# Process the first half of the image
midline = annotation.shape[-1] // 2
combined_regions_hemisphere_bin = combined_regions_bin.copy()
combined_regions_hemisphere_bin[:, :, midline:] = 0

# Set the binary values to 255
combined_regions_hemisphere_bin[combined_regions_hemisphere_bin == 1] = 255
combined_regions_bin[combined_regions_bin == 1] = 255

# Save the binarized images
tifffile.imwrite(os.path.join(analysis_directory, "hemisphere_mask.tif"), combined_regions_hemisphere_bin)
tifffile.imwrite(os.path.join(analysis_directory, "whole_brain_mask.tif"), combined_regions_bin)
