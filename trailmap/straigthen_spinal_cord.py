import os

import numpy as np
import skimage.morphology as morph
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import tifffile
from scipy.interpolate import splprep, splev

working_directory = (r"/mnt/data/Grace/projectome/fab/080724_rfp_fab_3D_MTG_FULL.dir/"
                     r"CTLS Capture - Pos 1 8 [1] 3DMontage Complete-1726216444-719.imgdir")
merged_raw_data_path = os.path.join(working_directory, "merged_resampled_raw.tif")
merged_raw_data = tifffile.imread(merged_raw_data_path)

background = 200
binary_merged_raw_data = merged_raw_data > background
binary_merged_raw_data = binary_merged_raw_data.astype("uint8")*255
tifffile.imwrite(os.path.join(working_directory, "binary_merged_raw_data.tif"), binary_merged_raw_data)

# 1. Skeletonize the 3D binary mask
skeleton = morph.skeletonize(binary_merged_raw_data)

# 2. Extract non-zero points (centerline points)
centerline_points = np.array(np.where(skeleton)).T  # Shape (N, 3)

# 3. Fit a spline to the centerline points
tck, u = splprep(centerline_points.T, s=0)
new_centerline_points = np.array(splev(u, tck)).T  # Shape (M, 3)

# 4. Create a grid for the new straightened volume
straightened_mask = np.zeros_like(binary_merged_raw_data)

# Create a mapping from the original coordinates to straightened coordinates
for i in range(len(new_centerline_points) - 1):
    print(i)
    start_point = new_centerline_points[i]
    end_point = new_centerline_points[i + 1]

    # Vector defining the direction from start to end point
    direction = end_point - start_point
    length = np.linalg.norm(direction)
    if length == 0:
        continue
    direction /= length  # Normalize

    # Iterate through the original mask and map points to the straightened volume
    for j in range(len(centerline_points)):
        orig_point = centerline_points[j]
        offset = orig_point - start_point

        # Project the offset onto the direction
        proj_length = np.dot(offset, direction)
        proj_point = start_point + proj_length * direction

        # Check bounds before assigning
        if np.all(proj_point >= 0) and np.all(proj_point < straightened_mask.shape):
            straightened_mask[tuple(np.round(proj_point).astype(int))] = 1  # Map to binary mask

# Visualization (slicing through the volume)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(centerline_points[:, 0], centerline_points[:, 1], color="red", linewidth=1)
axs[0].imshow(np.max(binary_merged_raw_data, 0), cmap='gray')
axs[0].set_title('Original Mask (Slice)')
axs[1].imshow(np.max(straightened_mask, 0), cmap='gray')
axs[1].set_title('Straightened Mask (Slice)')
# plt.show()
plt.savefig(os.path.join(working_directory, f"straigthened_spinal_cord.png"), dpi=300)
plt.savefig(os.path.join(working_directory, f"straigthened_spinal_cord.svg"), dpi=300)
