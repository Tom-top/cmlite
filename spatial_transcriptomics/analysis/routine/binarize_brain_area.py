import os

import tifffile

# Binarize the entire annotation
ano_directory = r"U:\Users\TTO\spatial_transcriptomics\atlas_ressources"
ano_path = os.path.join(ano_directory, f"gubra_annotations_spatial.tif")

region_to_mask = 5892  # Fixme: Implement multiple region masking
ref = tifffile.imread(ano_path)
ref_bin = ref == region_to_mask
ref_bin = ref_bin.astype("uint16")
ref_bin[ref_bin == 1] = 5000

tifffile.imwrite(os.path.join(ano_directory, f"bin_{region_to_mask}.tif"), ref_bin)