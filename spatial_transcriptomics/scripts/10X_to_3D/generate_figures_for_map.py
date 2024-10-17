import os

import numpy as np
import tifffile

import utils.utils as ut

coronal_slices_to_save = [185, 230, 270, 300, 350, 380, 410, 460]

working_directory = r"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results\gene_expression"

map_directories = [os.path.join(working_directory, i) for i in os.listdir(working_directory)]
maps = [os.path.basename(i) for i in map_directories]

for map, map_directory in zip(maps, map_directories):
    result_directory = ut.create_dir(os.path.join(map_directory, "figures"))
    map_data = tifffile.imread(os.path.join(map_directory, f"heatmap_{map}_neurons_dynamic.tif"))
    for projection in [0, 1, 2]:
        projection_data = np.min(map_data, projection) * 255
        projection_data = projection_data.astype("uint8")
        tifffile.imwrite(os.path.join(result_directory, f"max_projection_{projection}.tif"),
                         projection_data)
    for coronal_slice in coronal_slices_to_save:
        coronal_slice_data = map_data[coronal_slice, :, :, :3] * 255
        coronal_slice_data = coronal_slice_data.astype("uint8")
        tifffile.imwrite(os.path.join(result_directory, f"{coronal_slice}_coronal.tif"),
                         coronal_slice_data)
