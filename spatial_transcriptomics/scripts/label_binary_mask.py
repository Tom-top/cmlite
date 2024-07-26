import os

import numpy as np
import tifffile
from skimage.morphology import ball
from scipy.ndimage import label
from scipy.ndimage import find_objects
from scipy.ndimage import binary_opening, binary_closing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from utils.utils import assign_random_colors

MAP_DIR = r"/default/path"  # PERSONAL
TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"hemisphere_mask.tif"))
threshold_size = 50  # Set your size threshold here (in voxels)

struct = ball(1)  # Define a ball-shaped structuring element
opened_mask = binary_opening(TISSUE_MASK, structure=struct)  # Apply opening
smoothed_mask = binary_closing(opened_mask, structure=struct)  # Apply closing
labeled_mask, num_features = label(smoothed_mask)  # Label the connected components
objects_slices = find_objects(labeled_mask)  # Find the bounding boxes of each labeled component
component_sizes = [(labeled_mask[obj] != 0).sum() for obj in objects_slices]  # Calculate the size of each component
large_components_mask = np.zeros_like(labeled_mask, dtype=bool)  # Filter out small components
for i, obj in enumerate(objects_slices):
    if component_sizes[i] > threshold_size:
        large_components_mask[obj] = labeled_mask[obj] == (i + 1)
filtered_labeled_mask, num_filtered_features = label(large_components_mask)  # Label the filtered components
smoothed_mask_path = os.path.join(MAP_DIR, 'smoothed_mask.tif')
smoothed_mask_bin = filtered_labeled_mask.copy()
smoothed_mask_bin[smoothed_mask_bin > 0] = 255
tifffile.imwrite(smoothed_mask_path,
                 smoothed_mask_bin.astype("uint8")) # Save the filtered labeled mask as a TIFF file
labeled_mask_path = os.path.join(MAP_DIR, 'labeled_mask.tif')
tifffile.imwrite(labeled_mask_path,
                 filtered_labeled_mask.astype(np.int32)) # Save the filtered labeled mask as a TIFF file

########################################################################################################################
# GENERATE IMAGES OF THE LABELS
########################################################################################################################

f_name = os.path.basename(labeled_mask_path)
new_f_name = "rgb_" + f_name.split(".")[0] + "." + f_name.split(".")[-1]
# Load the 16-bit grayscale image
grayscale_image = tifffile.imread(labeled_mask_path).astype("uint16")
# Ensure the image is 16-bit
if grayscale_image.dtype != np.uint16:
    raise ValueError("The image is not 16-bit grayscale")
# Assign random colors to the grayscale image
colored_image = assign_random_colors(grayscale_image)
tifffile.imwrite(os.path.join(MAP_DIR, new_f_name), colored_image)

REFERENCE_FILE = r"resources/atlas/gubra_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)

ori = "horizontal"
orix, oriy = 2, 0
xlim, ylim = 369, 512

max_colored_image = np.max(colored_image, 1)
# Assuming colored_image is your image with shape (512, 369, 3)
height, width, channels = max_colored_image.shape
# Calculate the midpoint of the width
midpoint = width // 2
# Mirror the first half to the second half
mirrored_image = max_colored_image.copy()
mirrored_image[:, midpoint+1:] = max_colored_image[:, :midpoint][:, ::-1]
# Create an RGBA version of colored_image
colored_image_rgba = np.zeros((mirrored_image.shape[0], mirrored_image.shape[1], 4), dtype=np.float32)
# Set the RGB channels
colored_image_rgba[..., :3] = mirrored_image[..., :3] / mirrored_image.max()  # normalize RGB values if needed
# Set the alpha channel: 0 where colored_image is 0, 1 otherwise
colored_image_rgba[..., 3] = np.where(np.max(mirrored_image, axis=2) > 0, 1.0, 0.0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.rot90(np.max(REFERENCE, axis=orix))[::-1], cmap='gray_r', alpha=0.3)  # Display the reference image
ax.imshow(colored_image_rgba)  # Display the RGBA colored image
ax.set_xlim(0, xlim)
ax.set_ylim(0, ylim)
ax.invert_yaxis()
ax.axis('off')
fig.savefig(os.path.join(MAP_DIR, "rgb_" + f_name.split(".")[0] + ".png"), dpi=600)  # Save the figure
