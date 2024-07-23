import os

import nibabel as nib
import tifffile
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
# import pyclesperanto_prototype as cle


working_dir = "/home/grace/Documents/spatial_transcriptomics/Yao_et_al/processed_data/hotspot_detection_test"
z_score_path = os.path.join(working_dir, "zscore_1.nii.gz")

def smooth_3d_mask(binary_mask, sigma=10):
    smoothed_mask = gaussian_filter(binary_mask.astype(float), sigma=sigma)
    smoothed_binary_mask = (smoothed_mask > 0.5).astype(int)
    return smoothed_binary_mask

zscore = nib.load(z_score_path)
zscore_data = zscore.get_fdata()

thresh_v = 3
zscore_data_thresh = zscore_data.copy()
zscore_data_thresh[zscore_data_thresh <= thresh_v] = 0
zscore_data_bin = zscore_data_thresh.copy()
zscore_data_bin[zscore_data_bin > 0] = 1
zscore_data_bin = zscore_data_bin.astype("uint8")
zscore_data_smooth = smooth_3d_mask(zscore_data_bin, sigma=3)

fig = plt.figure(figsize=(20, 10))
ax0 = plt.subplot(121)
ax0.imshow(np.max(zscore_data_bin, 2), cmap="Greys_r")
ax1 = plt.subplot(122)
ax1.imshow(np.max(zscore_data_smooth, 2), cmap="Greys_r")

tifffile.imwrite(os.path.join(working_dir, "zscore_data_bin.tiff"), zscore_data_bin.astype("uint8"))
tifffile.imwrite(os.path.join(working_dir, "zscore_data_smooth.tiff"), zscore_data_smooth.astype("uint8"))

segmented = cle.voronoi_otsu_labeling(zscore_data_smooth, spot_sigma=3, outline_sigma=1)
cle.show(segmented, labels=True)









