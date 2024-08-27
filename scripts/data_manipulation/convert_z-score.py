import os

import nibabel as nib
import numpy as np
from natsort import natsorted
import tifffile

saving_dir = r"E:\AHA"  # Personal
gubra_atlas_tif_path = r"U:\BRAIN_ATLAS_TOOLS\gubra_atlas\version6_23sep2021_25um\tif_stack"
gubra_atlas = []

for plane in natsorted(os.listdir(gubra_atlas_tif_path)):
    z_plane = tifffile.imread(os.path.join(gubra_atlas_tif_path, plane))
    gubra_atlas.append(z_plane)

gubra_atlas = np.array(gubra_atlas)
tifffile.imwrite(os.path.join(saving_dir, "gubra_atlas.tif"), np.array(gubra_atlas))

g1, g2 = "g001", "g002"
z_score_path = fr"U:\2024\24-KIANG-0206\bifrost_results\642_cfos_unet\volumes\{g2}_against_{g1}_zscore.nii.gz"

z_score = nib.load(z_score_path).get_fdata()
z_score_positive = z_score.copy()
z_score_positive[z_score <= 0] = 0

z_score_img = np.array(z_score_positive).astype("uint16")
z_score_img = np.swapaxes(z_score_img, 0, 2)
tifffile.imwrite(os.path.join(saving_dir, f"z-score_{g2}_vs_{g1}.tif"), z_score_img)
