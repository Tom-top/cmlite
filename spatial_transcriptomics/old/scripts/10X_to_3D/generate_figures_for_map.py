import os

import numpy as np
import tifffile

import utils.utils as ut

# coronal_slices_to_save = [185, 230, 270, 300, 350, 380, 410, 460]
# coronal_slices_to_save = [262, 378, 388, 398, 428, 456]
coronal_slices_to_save = np.arange(350, 440, 5)

working_directory = r"/mnt/data/Thomas/whole_brain/results/gene_expression"

# target_genes = ["Chat_ENSMUSG00000021919", "Drd1_ENSMUSG00000021478", "Drd2_ENSMUSG00000032259",
#                 "Slc17a6_ENSMUSG00000030500", "Slc32a1_ENSMUSG00000037771", "Vsx2_ENSMUSG00000021239"]
target_genes = ["Chat_ENSMUSG00000021919", "Drd1_ENSMUSG00000021478", "Drd2_ENSMUSG00000032259"]
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
        for map_type in ["dynamic", "dynamic_raw", "dynamic_111_raw", "fixed"]:
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

atlas = tifffile.imread(os.path.join("/home/imaging/PycharmProjects/cmlite/resources/atlas/atlas_rgb",
                                     "gubra_annotation_mouse_rgb.tif"))
atlas = np.transpose(atlas, (1, 2, 0, 3))
for coronal_slice in coronal_slices_to_save:
    coronal_slice_atlas = atlas[coronal_slice, :, :, :3]
    coronal_slice_atlas = coronal_slice_atlas.astype("uint8")
    tifffile.imwrite(os.path.join(working_directory, f"{coronal_slice}_coronal_atlas_rgb.tif"),
                     coronal_slice_atlas)

atlas_16b = tifffile.imread(os.path.join("/home/imaging/PycharmProjects/cmlite/resources/atlas",
                                     "gubra_annotation_mouse.tif"))
atlas_16b = np.transpose(atlas_16b, (1, 2, 0))
for coronal_slice in coronal_slices_to_save:
    coronal_slice_atlas = atlas_16b[coronal_slice, :, :]
    tifffile.imwrite(os.path.join(working_directory, f"{coronal_slice}_16b_coronal_atlas.tif"),
                     coronal_slice_atlas)

region_mask = tifffile.imread(os.path.join("/mnt/data/Thomas/PPN_CUN",
                                     "whole_brain_mask_CUN.tif"))
for coronal_slice in coronal_slices_to_save:
    coronal_slice_atlas = region_mask[coronal_slice, :, :]
    tifffile.imwrite(os.path.join(working_directory, f"{coronal_slice}_CUN_mask.tif"),
                     coronal_slice_atlas)

