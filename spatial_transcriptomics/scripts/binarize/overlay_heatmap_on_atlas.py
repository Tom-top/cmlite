import os
import numpy as np
import tifffile
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import colormap module

########################################################################################################################
# GENERATE IMAGES OF THE MASK
########################################################################################################################

analysis_directory = r"E:\tto\GUS2022-189-LY"
mask_path = os.path.join(os.path.join(analysis_directory, "g006_against_g001_zscore_full_signal_threshold_3.nii.gz"))
f_name = os.path.basename(mask_path)
new_f_name = "rgb_" + f_name.split(".")[0] + "." + f_name.split(".")[-1]
# Load the 16-bit grayscale image
grayscale_image = nib.load(mask_path).get_fdata()
grayscale_image = np.swapaxes(grayscale_image, 0, 2)
grayscale_image = np.swapaxes(grayscale_image, 0, 1)

REFERENCE_FILE = r"resources/atlas/gubra_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)

ori = "horizontal"
orix, oriy = 2, 0
xlim, ylim = 369, 512

# Clip the grayscale image to remove low values and set transparency gradient
clip_min = 1.96  # Set the minimum value to retain
adjust_max = 1.96
clipped_grayscale_image = np.clip(grayscale_image, clip_min, None)

# Normalize the clipped grayscale image to range [0, 1] for proper display
normalized_grayscale_image = (clipped_grayscale_image - adjust_max) / (np.max(clipped_grayscale_image) - adjust_max)

# Apply a colormap to the normalized grayscale image
colormap = matplotlib.colormaps['viridis']  # You can choose any colormap here
colored_image = colormap(normalized_grayscale_image)

# Create an RGBA image with the colormap applied
rgba_image = np.zeros_like(colored_image, dtype=np.float32)
rgba_image[..., :3] = colored_image[..., :3]  # Copy the RGB channels from the colormap output

# Specify the range of grayscale values for the alpha gradient
alpha_min = 1.96  # Lower bound of the range
alpha_max = grayscale_image.max()  # Upper bound of the range

# Create a gradient for the alpha channel based on the specified range
alpha_gradient = np.zeros_like(grayscale_image, dtype=np.float32)
within_range = (grayscale_image >= alpha_min) & (grayscale_image <= alpha_max)
alpha_gradient[within_range] = (grayscale_image[within_range] - alpha_min) / (alpha_max - alpha_min)
alpha_gradient[grayscale_image > alpha_max] = 1
alpha_gradient = np.clip(alpha_gradient, 0, 1)

# Apply the alpha gradient to the RGBA image
rgba_image[..., 3] = alpha_gradient

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.rot90(np.max(REFERENCE, axis=orix))[::-1], cmap='gray_r', alpha=0.3)  # Display the reference image
ax.imshow(np.max(rgba_image, axis=1))  # Overlay the RGBA image with the colormap applied
ax.set_xlim(0, xlim)
ax.set_ylim(0, ylim)
ax.invert_yaxis()
ax.axis('off')
fig.savefig(os.path.join(analysis_directory, "rgb_" + f_name.split(".")[0] + ".png"), dpi=600)  # Save the figure
