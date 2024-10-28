import os

import numpy as np
import tifffile

import utils.utils as ut

coronal_slices_to_save = [185, 230, 270, 300, 350, 380, 410, 460]

working_directory = r"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results\gene_expression"

target_genes = ["Drd1"]
map_directories = [os.path.join(working_directory, i) for i in os.listdir(working_directory)
                   if i in target_genes]
if not isinstance(map_directories, list):
    map_directories = list(map_directories)
maps = [os.path.basename(i) for i in map_directories]

data_scaling = ["linear", "log2"]

for map, map_directory in zip(maps, map_directories):
    for scaling in data_scaling:
        scaling_directory = os.path.join(map_directory, scaling)
        result_directory = ut.create_dir(os.path.join(scaling_directory, "figures"))
        for map_type in ["dynamic", "dynamic_raw", "dynamic_111_raw"]:
            map_data = tifffile.imread(os.path.join(scaling_directory, f"heatmap_{map}_neurons_{map_type}.tif"))
            for projection in [0, 1, 2]:
                if map_type == "dynamic_raw_bin":
                    projection_data = np.max(map_data, projection) * 255
                else:
                    projection_data = np.min(map_data, projection) * 255
                projection_data = projection_data.astype("uint8")
                tifffile.imwrite(os.path.join(result_directory, f"max_projection_{projection}_{map_type}.tif"),
                                 projection_data)
            for coronal_slice in coronal_slices_to_save:
                if map_type == "dynamic_raw_bin":
                    coronal_slice_data = map_data[coronal_slice, :, :] * 255
                else:
                    coronal_slice_data = map_data[coronal_slice, :, :, :3] * 255
                coronal_slice_data = coronal_slice_data.astype("uint8")
                tifffile.imwrite(os.path.join(result_directory, f"{coronal_slice}_coronal_{map_type}.tif"),
                                 coronal_slice_data)
