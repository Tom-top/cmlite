import os

import numpy as np
import h5py
import tifffile

working_directory = r"E:\tto\bruker_troubleshooting\2024-07-10_130326_dual_illumination\raw\stack_2_channel_2_obj_bottom"

source_path = os.path.join(working_directory, "Cam_bottom_00000.lux.h5")
sink_path = os.path.join(working_directory, "Cam_bottom_00000.tif")

with h5py.File(source_path, "r") as f:
    data = f["Data"]
    tifffile.imwrite(sink_path, data)