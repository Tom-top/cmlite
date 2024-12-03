import os
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut

# Working directory and file paths
working_directory = (r"/mnt/data/Grace/projectome/1_earmark_gfp_20tiles_2z_3p25zoom_singleill/raw/sample_1/skeletonization")
saving_directory = ut.create_dir(os.path.join(working_directory, "figures"))

merged_weighted_skeleton_path = os.path.join(working_directory, "weighted_skeleton_clean.tif")
merged_binary_skeleton_path = os.path.join(working_directory, "binarized_skeleton_clean.tif")
merged_weighted_skeleton = tifffile.imread(merged_weighted_skeleton_path)
merged_binary_skeleton = tifffile.imread(merged_binary_skeleton_path)

########################################################################################################################
# CREATE HISTOGRAMS FOR the ROSTRO-CAUDAL VIEW
########################################################################################################################

# Sum along each dimension for histograms
x_projection = np.sum(merged_binary_skeleton, axis=(0, 1))  # Sum across Y and Z
y_projection = np.sum(merged_binary_skeleton, axis=(0, 2))  # Sum across X and Z

# Max projection for X dimension
x_max_projection = np.sum(merged_weighted_skeleton, axis=0)  # Max projection across X (YZ plane)

# Plot the X max projection with the X and Y histograms
fig, axes = plt.subplots(2, 2, figsize=(10, 15))

# X dimension: Max projection
middle_x = len(x_projection) / 2
axes[0, 0].imshow(x_max_projection, cmap='gray')
axes[0, 0].set_title('Max Projection (YZ plane - X axis)')
axes[0, 0].axis('off')
axes[0, 0].axvline(middle_x, color='red', linestyle='--', label='Midline')

# Get X and Y-axis limits from the max projection image (to match histogram dimensions)
xlim_x_projection = axes[0, 0].get_xlim()  # This returns the X-axis limits (width of the image)
ylim_x_projection = axes[0, 0].get_ylim()  # This returns the Y-axis limits (height of the image)

# Calculate the rolling average for X projection
window_size = 20  # Set the window size
x_rolling_avg = pd.Series(x_projection).rolling(window=window_size, center=True).mean()

# X dimension histogram
axes[1, 0].bar(range(len(x_projection)), x_projection, color="gray")
axes[1, 0].set_title('Distribution of Positive Voxels Across X Dimension')
axes[1, 0].set_xlabel('X Axis')
axes[1, 0].set_ylabel('Positive Voxel Count')
# Match X histogram X-axis to the max projection Y-axis
axes[1, 0].set_xlim(xlim_x_projection)  # Adjust the X-axis of the histogram to match the Y-dimension of the image

# Plot the rolling average curve for the X projection
axes[1, 0].plot(range(len(x_projection)), x_rolling_avg, color='gray', linestyle='-', label='Rolling Avg')

# Add dashed line to separate histogram into two parts
axes[1, 0].axvline(middle_x, color='black', linestyle='--', label='Midline')
axes[1, 0].legend()

# Calculate the rolling average for Y projection
y_rolling_avg = pd.Series(y_projection).rolling(window=window_size, center=True).mean()

# Y dimension histogram (rotated 90 degrees)
axes[0, 1].barh(range(len(y_projection)), y_projection, color="gray")
axes[0, 1].set_title('Distribution of Positive Voxels Across Y Dimension')
axes[0, 1].set_xlabel('Positive Voxel Count')
axes[0, 1].set_ylabel('Y Axis')
# Invert the Y-axis for the horizontal bar chart to align with the image
axes[0, 1].invert_yaxis()
# Match Y histogram Y-axis to the max projection's height
axes[0, 1].set_ylim(ylim_x_projection)  # Match the Y-axis of the histogram to the image

# Plot the rolling average curve for the Y projection
axes[0, 1].plot(y_rolling_avg, range(len(y_projection)), color='gray', linestyle='-', label='Rolling Avg')

# Hide the unused subplot
axes[1, 1].axis('off')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(saving_directory, "rostro_caudal_histograms.png"), dpi=300)
plt.savefig(os.path.join(saving_directory, "rostro_caudal_histograms.svg"), dpi=300)


########################################################################################################################
# CREATE HISTOGRAMS FOR CORONAL CHUNKS
########################################################################################################################
#
# chunk_size = 100
# chunks = np.arange(0, merged_weighted_skeleton.shape[1], chunk_size)
#
# for n, chunk in enumerate(chunks):
#
#     chunk_data = merged_weighted_skeleton[:, chunk:chunk+chunk_size, :]
#
#     # Sum along each dimension for histograms
#     x_projection = np.sum(chunk_data, axis=(0, 1))  # Sum across Y and Z
#     z_projection = np.sum(chunk_data, axis=(1, 2))  # Sum across X and Y
#
#     # Max projection for X dimension
#     y_max_projection = np.sum(chunk_data, axis=1)  # Max projection across Y (XZ plane)
#     y_max_projection_shape = y_max_projection.shape
#     aspect_ratio = y_max_projection_shape[0]/y_max_projection_shape[1]
#     aspect_ratio = (1100-600)/y_max_projection_shape[1]
#
#     # Plot the X max projection with the X and Y histograms
#     fig, axes = plt.subplots(2, 2, figsize=(10, 10*aspect_ratio))
#
#     # X dimension: Max projection
#     middle_x = len(x_projection) / 2
#     axes[0, 0].imshow(y_max_projection, cmap='gray')
#     axes[0, 0].set_title('Max Projection (YZ plane - X axis)')
#     axes[0, 0].axis('off')
#     axes[0, 0].axvline(middle_x, color='red', linestyle='--', label='Midline')
#     axes[0, 0].set_ylim(1100, 600)
#
#     # Get X and Y-axis limits from the max projection image (to match histogram dimensions)
#     xlim_x_projection = axes[0, 0].get_xlim()  # This returns the X-axis limits (width of the image)
#     ylim_x_projection = axes[0, 0].get_ylim()  # This returns the Y-axis limits (height of the image)
#
#     # X dimension histogram
#     axes[1, 0].bar(range(len(x_projection)), x_projection, color="gray")
#     axes[1, 0].set_title('Distribution of Positive Voxels Across X Dimension')
#     axes[1, 0].set_xlabel('X Axis')
#     axes[1, 0].set_ylabel('Positive Voxel Count')
#     # Match X histogram X-axis to the max projection Y-axis
#     axes[1, 0].set_xlim(xlim_x_projection)  # Adjust the X-axis of the histogram to match the Y-dimension of the image
#
#     # Add dashed line to separate histogram into two parts
#     axes[1, 0].axvline(middle_x, color='black', linestyle='--', label='Midline')
#     axes[1, 0].legend()
#
#     # Y dimension histogram (rotated 90 degrees)
#     axes[0, 1].barh(range(len(z_projection)), z_projection, color="gray")
#     axes[0, 1].set_title('Distribution of Positive Voxels Across Y Dimension')
#     axes[0, 1].set_xlabel('Positive Voxel Count')
#     axes[0, 1].set_ylabel('Y Axis')
#     # Invert the Y-axis for the horizontal bar chart to align with the image
#     axes[0, 1].invert_yaxis()
#     # Match Y histogram Y-axis to the max projection's height
#     # axes[0, 1].set_ylim(ylim_x_projection)  # Match the Y-axis of the histogram to the image
#     axes[0, 1].set_ylim(1100, 600)
#
#     # Hide the unused subplot
#     axes[1, 1].axis('off')
#
#     # Adjust layout and save the figure
#     plt.tight_layout()
#     plt.savefig(os.path.join(saving_directory, f"chunk_{n}_histograms.png"), dpi=300)
#     plt.savefig(os.path.join(saving_directory, f"chunk_{n}_histograms.svg"), dpi=300)
