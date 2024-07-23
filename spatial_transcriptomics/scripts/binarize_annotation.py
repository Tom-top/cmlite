import os
import numpy as np
import tifffile

import utils.utils as ut

# Define paths and cutoff value for binarization
ref_directory = r"resources\atlas"
ref_path = os.path.join(ref_directory, "gubra_annotation_mouse.tif")
analysis_directory = ut.create_dir(r"E:\tto\spatial_transcriptomics_results\whole_brain")  # PERSONAL
cutoff = 0  # Value above which all pixels will be kept for the mask. 0 = everything but background

# Load and transpose the image
annotation = np.transpose(tifffile.imread(ref_path), (1, 2, 0))

# Binarize the image based on the cutoff
annotation_bin = (annotation > cutoff).astype("uint8")

# Process the first half of the image
midline = annotation.shape[-1] // 2
annotation_hemisphere_bin = annotation_bin[:, :, :midline]

# Set the binary values to 255
annotation_hemisphere_bin[annotation_hemisphere_bin == 1] = 255
annotation_bin[annotation_bin == 1] = 255

# Save the binarized images
tifffile.imwrite(os.path.join(analysis_directory, "hemisphere_mask.tif"), annotation_hemisphere_bin)
tifffile.imwrite(os.path.join(analysis_directory, "whole_brain_mask.tif"), annotation_bin)
