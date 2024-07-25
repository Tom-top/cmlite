import os

import numpy as np
import tifffile
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut

# Define paths and cutoff value for binarization
zscore_maps_directory = r"/default/path"  # PERSONAL
zscore_map_name = "Semaglutide"
zscore_map_directory = os.path.join(zscore_maps_directory, zscore_map_name)
zscore_map_path = os.path.join(zscore_map_directory, "result.nii.gz")
analysis_directory = ut.create_dir(fr"/default/path")  # PERSONAL
cutoff = 320  # Value above which all pixels will be kept for the mask (z-score * 100)

# Load and transpose the image
zscore_map_nii = nib.load(zscore_map_path)
zscore_map = zscore_map_nii.get_fdata()
zscore_map = np.flip(np.transpose(zscore_map, (1, 2, 0)), 1)

# Binarize the image based on the cutoff
zscore_map_bin = (zscore_map >= cutoff).astype("uint8")

# Process the first half of the image
midline = zscore_map.shape[-1] // 2
zscore_map_bin_half = zscore_map_bin.copy()
zscore_map_bin_half[:, :, midline:] = 0

# Set the binary values to 255
zscore_map_bin[zscore_map_bin == 1] = 255
zscore_map_bin_half[zscore_map_bin_half == 1] = 255

# Save the binarized images
tifffile.imwrite(os.path.join(analysis_directory, "hemisphere_mask.tif"), zscore_map_bin_half)
tifffile.imwrite(os.path.join(analysis_directory, "whole_brain_mask.tif"), zscore_map_bin)

########################################################################################################################
# GENERATE IMAGES OF THE MASK
########################################################################################################################

mask_path = os.path.join(analysis_directory, "whole_brain_mask.tif")
f_name = os.path.basename(mask_path)
new_f_name = "rgb_" + f_name.split(".")[0] + "." + f_name.split(".")[-1]
# Load the 16-bit grayscale image
grayscale_image = tifffile.imread(mask_path).astype("uint16")
grayscale_image = grayscale_image / 10

REFERENCE_FILE = r"resources/atlas/gubra_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)

ori = "horizontal"
orix, oriy = 2, 0
xlim, ylim = 369, 512

colored_image = np.stack((grayscale_image,)*3, axis=-1)
max_image = np.max(colored_image, 1)
# Create an RGBA version of colored_image
colored_image_rgba = np.zeros((max_image.shape[0], max_image.shape[1], 4), dtype=np.float32)
# Set the alpha channel: 0 where colored_image is 0, 1 otherwise
colored_image_rgba[..., 3] = np.where(np.max(max_image, axis=2) > 0, 1.0, 0.0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.rot90(np.max(REFERENCE, axis=orix))[::-1], cmap='gray_r', alpha=0.3)  # Display the reference image
ax.imshow(colored_image_rgba)  # Display the RGBA colored image
ax.set_xlim(0, xlim)
ax.set_ylim(0, ylim)
ax.invert_yaxis()
ax.axis('off')
fig.savefig(os.path.join(analysis_directory, "rgb_" + f_name.split(".")[0] + ".png"), dpi=600)  # Save the figure
