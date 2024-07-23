import os

import tifffile

import utils.utils as ut

zscore_map_directory = r"/mnt/data/spatial_transcriptomics/results/mpd5_pick1"  # PERSONAL
zscore_map_name = "Pick-1_vs_Vehicle_pval_up_coronal"
zscore_map_path = os.path.join(zscore_map_directory, f"{zscore_map_name}.tif")

analysis_directory = ut.create_dir("/mnt/data/Thomas/test")  # PERSONAL

cutoff = 80

zscore_map = tifffile.imread(zscore_map_path)
zscore_map_bin = zscore_map >= cutoff
zscore_map_bin_half = zscore_map_bin[:, :, :int(369/2)]
zscore_map_bin_half = zscore_map_bin_half.astype("uint8")
zscore_map_bin_half[zscore_map_bin_half == 1] = 255
tifffile.imwrite(os.path.join(analysis_directory, "hemisphere_mask.tif"), ref_hemisphere_bin)
tifffile.imwrite(os.path.join(analysis_directory, "whole_brain_mask.tif"), ref_whole_brain_bin)
