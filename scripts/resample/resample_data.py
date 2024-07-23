import os

import resampling.resampling as res

working_directory = r""  # PERSONAL
converted_file = os.path.join(working_directory, "stitched_6.npy")

resample_5um_parameter = {
    "source_resolution": (5, 1.46, 1.46),
    "sink_resolution": (5, 5, 5),
    "processes": None,
    "verbose": True,
    "method": "memmap",
}

resampled_5_path = os.path.join(working_directory, "stitched_6_5um.tif")
res.resample(converted_file, sink=resampled_5_path, **resample_5um_parameter)
