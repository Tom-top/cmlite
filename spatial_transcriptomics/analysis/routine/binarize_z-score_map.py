import os

import tifffile

zscore_map_directory = r"/mnt/data/spatial_transcriptomics/results/mpd5_pick1"
zscore_map_name = "Pick-1_vs_Vehicle_pval_up_coronal"
zscore_map_path = os.path.join(zscore_map_directory, f"{zscore_map_name}.tif")

cutoff = 80

zscore_map = tifffile.imread(zscore_map_path)
zscore_map_bin = zscore_map >= cutoff
zscore_map_bin_half = zscore_map_bin[:, :, :int(369/2)]
zscore_map_bin_half = zscore_map_bin_half.astype("uint8")
zscore_map_bin_half[zscore_map_bin_half == 1] = 255
tifffile.imwrite(os.path.join(zscore_map_directory, f"{zscore_map_name}_bin.tif"), zscore_map_bin_half)
tifffile.imwrite(os.path.join(zscore_map_directory, f"{zscore_map_name}_bin_whole.tif"), zscore_map_bin)
