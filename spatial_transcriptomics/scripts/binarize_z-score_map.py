import os

import numpy as np
import tifffile
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut
import spatial_transcriptomics.utils.utils as sut

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

ATLAS_USED = "gubra"
ANO_DIRECTORY = r"resources\atlas"
REFERENCE_FILE = os.path.join(ANO_DIRECTORY, fr"{ATLAS_USED}_reference_mouse.tif")
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

########################################################################################################################
# GENERATE RGB IMAGE OF THE MASK
########################################################################################################################

ANNOTATION_FILE = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.tif")
ANO = np.transpose(tifffile.imread(ANNOTATION_FILE), (1, 2, 0))
ANO_JSON = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.json")
metadata = ut.load_json_file(ANO_JSON)
unique_ids = np.unique(ANO)
n_unique_ids = len(unique_ids)

aba_colored_mask = colored_image.copy() / 255

for n, uid in enumerate(unique_ids):
    if uid not in [0, 5000]:
        uid_color = sut.find_dict_by_key_value(metadata, uid)["color_hex_triplet"]
        uid_name = sut.find_dict_by_key_value(metadata, uid)["acronym"]
        if uid_color is not None:
            ut.print_c(f"[INFO] Coloring mask for region: {uid_name}; {n+1}/{n_unique_ids}")
            # uid_color_hex = np.array([i * 255 for i in ut.hex_to_rgb("#" + uid_color)])
            uid_color_hex = ut.hex_to_rgb("#" + uid_color)
            uid_mask = ANO == uid
            mask_intersections = np.logical_and(uid_mask, grayscale_image)
            aba_colored_mask[mask_intersections] = uid_color_hex

def project_first_nonzero(aba_colored_mask, axis, direction="forward"):
    # Move the chosen axis to the front
    aba_colored_mask = np.moveaxis(aba_colored_mask, axis, 0)
    depth, height, width, channels = aba_colored_mask.shape

    first_nonzero_image = np.zeros((height, width, channels), dtype=aba_colored_mask.dtype)

    # Determine the range to iterate over based on the direction
    if direction == "forward":
        range_iter = range(depth)  # Project from up to down or front to back
    elif direction == "backward":
        range_iter = range(depth - 1, -1, -1)  # Project from down to up or back to front

    # Iterate over the chosen axis in the specified direction
    for i in range_iter:
        mask = np.any(first_nonzero_image == 0, axis=2) & np.any(aba_colored_mask[i] > 0, axis=2)
        first_nonzero_image[mask] = aba_colored_mask[i][mask]

    return first_nonzero_image

# Choose the axis over which to project (0 for depth, 1 for height, 2 for width)
axis = 1  # Example: choose depth
direction = "backward"  # Choose "forward" or "backward" for projection direction
first_nonzero_image = project_first_nonzero(aba_colored_mask, axis, direction)

# Create an RGBA version of the first_nonzero_image
colored_image_rgba = np.zeros((*first_nonzero_image.shape[:2], 4), dtype=np.float32)
colored_image_rgba[..., :3] = first_nonzero_image
colored_image_rgba[..., 3] = np.where(np.any(first_nonzero_image > 0, axis=2), 1.0, 0.0)

# Plot the result
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.rot90(np.max(REFERENCE, axis=orix))[::-1], cmap='gray_r', alpha=0.3)
ax.imshow(colored_image_rgba)
ax.set_xlim(0, xlim)
ax.set_ylim(0, ylim)
ax.invert_yaxis()
ax.axis('off')

# Save the figure
fig.savefig(os.path.join(analysis_directory, "aba_rgb_" + f_name.split(".")[0] + "_first_nonzero.png"), dpi=600)
