import os

import resampling.resampling as res

working_directory = r"/default/path"  # PERSONAL

# REFERENCE

reference_path = os.path.join(working_directory, "gubra_reference_mouse.tif")
reference_10um_path = os.path.join(working_directory, "gubra_reference_mouse_10um.tif")

resample_10um_parameter = {
    "source_resolution": (25, 25, 25),
    "sink_resolution": (10, 10, 10),
    "processes": None,
    "verbose": True,
    "method": "memmap",
}

res.resample(reference_path, sink=reference_10um_path, **resample_10um_parameter)

# REFERENCE NO BULBS

reference_path = os.path.join(working_directory, "gubra_reference_nb_mouse.tif")
reference_10um_path = os.path.join(working_directory, "gubra_reference_nb_mouse_10um.tif")

resample_10um_parameter = {
    "source_resolution": (25, 25, 25),
    "sink_resolution": (10, 10, 10),
    "processes": None,
    "verbose": True,
    "method": "memmap",
}

res.resample(reference_path, sink=reference_10um_path, **resample_10um_parameter)

# ATLAS

atlas_path = os.path.join(working_directory, "gubra_annotation_mouse.tif")
atlas_10um_path = os.path.join(working_directory, "gubra_annotation_mouse_10um.tif")

resample_10um_parameter = {
    "source_resolution": (25, 25, 25),
    "sink_resolution": (10, 10, 10),
    "processes": None,
    "verbose": True,
    "method": "memmap",
    "interpolation": 'nearest'
}

res.resample(atlas_path, sink=atlas_10um_path, **resample_10um_parameter)
