import os

import cv2
import SimpleITK as sitk

import resampling.resampling as res

working_directory = r"/default/path"  # PERSONAL
converted_file = os.path.join(working_directory, "result.mhd")
atlas_to_auto = sitk.ReadImage(converted_file)  # Load the .mhd image
atlas_to_auto_array = sitk.GetArrayFromImage(atlas_to_auto)  # Convert to numpy array

resample_5um_parameter = {
    "source_resolution": (25, 25, 25),
    "sink_resolution": (5, 5, 5),
    "sink_shape": (2024, 2676, 1128),
    "processes": None,
    "verbose": True,
    "method": "memmap",  # memmape; shared
    "interpolation": "nearest",  # nearest = no interpolation (for atlas); linear otherwise
}

resampled_5_path = os.path.join(working_directory, "result_5um.tif")
res.resample(atlas_to_auto_array, sink=resampled_5_path, **resample_5um_parameter)
