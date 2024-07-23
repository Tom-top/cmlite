import os

import tifffile

# Binarize the entire annotation
ref_directory = r"/mnt/data/spatial_transcriptomics/results/whole_brain"  # PERSONAL
ref_path = os.path.join(ref_directory, f"atlas_annotations_coronal.tif")

cutoff = 0
ref = tifffile.imread(ref_path)
ref_bin = ref > cutoff
ref_bin_half = ref_bin[:, :, :int(369/2)]
ref_bin_half = ref_bin_half.astype("uint8")
ref_bin_half[ref_bin_half == 1] = 255
ref_bin = ref_bin.astype("uint8")
ref_bin[ref_bin == 1] = 255

tifffile.imwrite(os.path.join(ref_directory, f"bin.tif"), ref_bin_half)
tifffile.imwrite(os.path.join(ref_directory, f"bin_whole.tif"), ref_bin)