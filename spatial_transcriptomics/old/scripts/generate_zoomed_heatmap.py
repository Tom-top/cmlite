import os

import numpy as np
import nibabel as nib
import tifffile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

import spatial_transcriptomics.old.utils.plotting as st_plt


REFERENCE_FILE = r"resources/atlas/gubra_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)

MAP_DIR = r"/default/path"  # PERSONAL
TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"hemisphere_mask.tif"))

HEATMAP_DIR = r"/resources/z-score_maps/Semaglutide"
HEATMAP = np.flip(nib.load(os.path.join(HEATMAP_DIR, "result.nii.gz")).get_fdata(), 2)
# HEATMAP = np.transpose(HEATMAP, (1, 2, 0))

ori = "coronal"
orix, oriy, mask_axis = 2, 1, 0  # Projection = 1
xlim, ylim = 369, 268

# tifffile.imwrite(os.path.join(r"E:\tto\spatial_transcriptomics_results\PB\results\3d_views", "hm_test.tif"), max_proj_heatmap)
# tifffile.imwrite(os.path.join(r"E:\tto\spatial_transcriptomics_results\PB\results\3d_views", "ref_test.tif"), max_proj_reference)

max_proj_mask = np.max(TISSUE_MASK, mask_axis)
sum_along_axis = np.sum(TISSUE_MASK, axis=tuple(np.delete(np.arange(0, 3), mask_axis)))

# Find the indices where the sum is greater than 0
positive_indices = np.where(sum_along_axis > 0)[0]

# The first and last indices with positive pixels
first_plane_with_positive_pixels = positive_indices[0]
last_plane_with_positive_pixels = positive_indices[-1]

max_proj_reference = np.rot90(np.max(REFERENCE[first_plane_with_positive_pixels:last_plane_with_positive_pixels], axis=oriy))[::-1]
max_proj_heatmap = np.rot90(np.max(HEATMAP[first_plane_with_positive_pixels:last_plane_with_positive_pixels], axis=oriy))[::-1]

if mask_axis == 2:
    max_proj_mask = np.swapaxes(max_proj_mask, 0, 1)
top_left, bottom_right = st_plt.find_square_bounding_box(max_proj_mask, 15)
cropped_ref = max_proj_reference[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
cropped_heatmap = max_proj_heatmap[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(max_proj_heatmap[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], cmap='cividis',
          alpha=0.3)
# ax.imshow(max_proj_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]],
#           alpha=1)
ax.set_xlim(0, cropped_ref.shape[0])
ax.set_ylim(0, cropped_ref.shape[-1])
ax.invert_yaxis()
ax.axis('off')
fig.savefig(os.path.join(os.path.dirname(saving_path), f"mask_{ori}_zoom.png"), dpi=300)
fig.savefig(os.path.join(os.path.dirname(saving_path), f"mask_{ori}_zoom.svg"), dpi=300)
plt.close(fig)
