import os

import tifffile
import skimage.io as skio

import utils.utils as ut

import resampling.resampling as res

import alignment.align as elx

working_directory = r"E:\LLL\MBP_data\raw\ID522_an000002_g002_Brain_M3_rescan2\xy5p0_z10p0\2024-09-01_173641_merged"

########################################################################################################################
# TRANSFORM ATLAS TO AUTO
########################################################################################################################

ref_to_auto_directory = os.path.join(working_directory, f"mouse_gubra_reference_to_auto_1")
atlas_path = r"C:\Users\MANDOUDNA\PycharmProjects\cmlite\resources\atlas\gubra_annotation_mouse_-1_2_3_None.tif"
result_dir = os.path.join(working_directory, "atlas_to_auto")
transform_atlas_parameter = dict(
    source=atlas_path,
    result_directory=result_dir,
    transform_parameter_file=os.path.join(ref_to_auto_directory, f"TransformParameters_ni.1.txt"))
elx.transform_images(**transform_atlas_parameter)

########################################################################################################################
# UPSCALE THE ATLAS
########################################################################################################################

transformed_atlas = skio.imread(os.path.join(result_dir, "result.mhd"), plugin='simpleitk')

resample_parameter = {
    "source_resolution": (25, 25, 25),
    "sink_resolution": (10, 10, 10),
    # "sink_shape": (),
    "processes": None,
    "verbose": True,
    "method": "memmap",
    "interpolation": "linear",  #nearest
}

res.resample(transformed_atlas, sink=os.path.join(result_dir, "upsampled_atlas_2.tif"), **resample_parameter)
