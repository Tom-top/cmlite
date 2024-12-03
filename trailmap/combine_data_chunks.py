import os

from natsort import natsorted
import numpy as np
import tifffile

import utils.utils as ut

working_directory = ("/mnt/data/Grace/projectome/1_earmark_gfp_20tiles_2z_3p25zoom_singleill/raw/sample_1")

chunk_dirs = natsorted([os.path.join(working_directory, i) for i in os.listdir(working_directory)
                        if i.startswith("chunk")])
n_chunks = len(chunk_dirs)

merged_skeleton = []

for n, chunk in enumerate(chunk_dirs):
    chunk_name = os.path.basename(chunk)
    ut.print_c(f"[INFO] Loading cleaned skeleton for chunk: {chunk_name}; {n+1}/{n_chunks}!")
    skeleton_path = os.path.join(chunk, "skeleton")
    # cleaned_skeleton_path = os.path.join(skeleton_path, "weighted_skeleton_clean.tif")
    # cleaned_skeleton_path = os.path.join(skeleton_path, "binarized_skeleton_clean.tif")
    cleaned_skeleton_path = os.path.join(chunk, "resampled_raw.tif")

    # model_path = os.path.join(chunk, "denardo_2_output_ch0")
    # cleaned_skeleton = [tifffile.imread(os.path.join(model_path, i)) for i in natsorted(os.listdir(model_path))]
    # cleaned_skeleton = np.array(cleaned_skeleton)
    # cleaned_skeleton = cleaned_skeleton - cleaned_skeleton.min() / (cleaned_skeleton.max() - cleaned_skeleton.min())
    # cleaned_skeleton = cleaned_skeleton * 255
    # cleaned_skeleton = cleaned_skeleton.astype("uint8")
    # tifffile.imwrite(os.path.join(chunk, "denardo_2_output_ch0.tif"), cleaned_skeleton)

    # cleaned_skeleton = tifffile.imread(os.path.join(chunk, "denardo_1_output_ch0.tif"))
    cleaned_skeleton = tifffile.imread(cleaned_skeleton_path)
    merged_skeleton.append(cleaned_skeleton)

# merged_skeleton = np.array(merged_skeleton)
merged_skeleton = np.hstack(merged_skeleton)
tifffile.imwrite(os.path.join(working_directory, "resampled_raw.tif"), merged_skeleton)

#
# merged_skeleton = tifffile.imread("/mnt/data/Grace/projectome/fab_redo/projectome_fab_3D_MTG_FULL.dir"
#                                   "/CTLS Capture - Pos 1 9 [1] 3DMontage Complete-1727093864-196.imgdir/binarized_skeleton_clean.tif")
# avg_merged_skeleton = np.mean(merged_skeleton, axis=0)
# tifffile.imwrite(os.path.join(working_directory, "mean_binarized_skeleton_clean.tif"), avg_merged_skeleton)
