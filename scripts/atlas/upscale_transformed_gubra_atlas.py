import os

import numpy as np
import skimage.io as skio
import tifffile

import resampling.resampling as res

working_directory = r"/default/path"  # PERSONAL

channels = [1, 3, 5]

for channel in channels:

    channel_path = os.path.join(working_directory, f"mouse_gubra_atlas_to_auto_{channel}")
    resampled_channel = tifffile.imread(os.path.join(working_directory, f"resampled_10um_{channel}.tif"))
    resampled_channel = np.swapaxes(resampled_channel, 0, 2)

    atlas_path = os.path.join(channel_path, "result.mhd")
    transformed_atlas = skio.imread(atlas_path, plugin='simpleitk')
    transformed_atlas = np.swapaxes(transformed_atlas, 0, 2)
    atlas_10um_path = os.path.join(channel_path, "result_10um.tif")

    resample_10um_parameter = {
        "source_resolution": (25, 25, 25),
        "sink_resolution": (10, 10, 10),
        "sink_shape": resampled_channel.shape,
        "processes": None,
        "verbose": True,
        "method": "memmap",
        "interpolation": 'nearest'
    }

    res.resample(transformed_atlas, sink=atlas_10um_path, **resample_10um_parameter)
