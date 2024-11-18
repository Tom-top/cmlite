"""
This script generates a binary mask of the desired annotation file:
Options:
- Gubra V6: gubra
- CCFV3: aba
"""

import os
import numpy as np
import tifffile
import utils.utils as ut

########################################################################################################################
# Setup Paths and Parameters
########################################################################################################################

# Specify the annotation atlas and directory paths
annotation_used = "aba"  # Options: 'gubra' or 'aba'
annotation_dir = r"resources\atlas"
annotation_path = os.path.join(annotation_dir, f"{annotation_used}_annotation_mouse.tif")

# Define directories for saving outputs
working_dir = r"E:\tto"
saving_dir = ut.create_dir(os.path.join(working_dir, annotation_used))  # Customized for user setup
mask_dir = ut.create_dir(os.path.join(saving_dir, "masks"))  # Directory to save mask files

# Binarization cutoff: pixels with values above this will be kept in the mask (0 keeps everything except the background)
cutoff = 0

########################################################################################################################
# Load and Binarize Annotation Image
########################################################################################################################

# Load the annotation image and transpose to (x, y, z) orientation for processing
annotation = np.transpose(tifffile.imread(annotation_path), (1, 2, 0))

# Create binary mask: pixels greater than cutoff are set to 1
annotation_bin = (annotation > cutoff).astype("uint8")

########################################################################################################################
# Process Hemisphere Mask
########################################################################################################################

# Generate a binary mask for one hemisphere by zeroing out the other half
midline = annotation.shape[-1] // 2  # Calculate midline for hemisphere separation
annotation_hemisphere_bin = annotation_bin.copy()
annotation_hemisphere_bin[:, :, midline:] = 0  # Zero out the right hemisphere

# Set binary values to 255 for saving as an 8-bit image
annotation_hemisphere_bin[annotation_hemisphere_bin == 1] = 255
annotation_bin[annotation_bin == 1] = 255

########################################################################################################################
# Save the Binarized Images
########################################################################################################################

# Save the binary masks for the whole brain and the hemisphere
tifffile.imwrite(os.path.join(mask_dir, "hemisphere_mask.tif"), annotation_hemisphere_bin)
tifffile.imwrite(os.path.join(mask_dir, "whole_brain_mask.tif"), annotation_bin)
