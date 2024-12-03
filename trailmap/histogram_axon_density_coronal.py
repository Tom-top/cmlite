import os
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut

# Working directory and file paths
working_directory = (r"/mnt/data/Grace/projectome/1_earmark_gfp_20tiles_2z_3p25zoom_singleill/raw/sample_1/skeletonization/figures")
saving_directory = ut.create_dir(os.path.join(working_directory, "histogram"))

start_end = "4900-4930"
merged_weighted_skeleton_path = os.path.join(working_directory, f"coronal_{start_end}_raw.tif")
merged_binary_skeleton_path = os.path.join(working_directory, f"coronal_{start_end}_skel_sum.tif")
merged_weighted_skeleton = tifffile.imread(merged_weighted_skeleton_path)
merged_binary_skeleton = tifffile.imread(merged_binary_skeleton_path)

########################################################################################################################
# CREATE HISTOGRAMS FOR the ROSTRO-CAUDAL VIEW
########################################################################################################################

# Sum along each dimension for histograms
x_projection = np.sum(merged_binary_skeleton, axis=0)  # Sum across Y and Z
y_projection = np.sum(merged_binary_skeleton, axis=1)  # Sum across X and Z

# Max projection for X dimension
x_max_projection = merged_weighted_skeleton  # Max projection across X (YZ plane)

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
plt.savefig(os.path.join(saving_directory, f"rostro_caudal_histograms_{start_end}.png"), dpi=300)
plt.savefig(os.path.join(saving_directory, f"rostro_caudal_histograms_{start_end}.svg"), dpi=300)
