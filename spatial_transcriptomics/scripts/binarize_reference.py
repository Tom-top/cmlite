import os
import numpy as np
import tifffile

import utils.utils as ut

# Binarize the entire annotation
ref_directory = r"/default/path"  # PERSONAL
ref_path = os.path.join(ref_directory, "gubra_annotation_mouse.tif")

analysis_directory = ut.create_dir("/default/path")  # PERSONAL

cutoff = 0

# Load and transpose the image
ref = tifffile.imread(ref_path)
ref = np.transpose(ref, (1, 2, 0))

# Binarize the image based on the cutoff
ref_whole_brain_bin = (ref > cutoff).astype("uint8")

# Process the first half of the image
midline = ref.shape[-1] // 2
ref_hemisphere_bin = ref_whole_brain_bin[:, :, :midline]

# Set the binary values to 255
ref_hemisphere_bin[ref_hemisphere_bin == 1] = 255
ref_whole_brain_bin[ref_whole_brain_bin == 1] = 255

# Save the binarized images
tifffile.imwrite(os.path.join(analysis_directory, "hemisphere_mask.tif"), ref_hemisphere_bin)
tifffile.imwrite(os.path.join(analysis_directory, "whole_brain_mask.tif"), ref_whole_brain_bin)
