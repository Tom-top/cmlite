import os

import numpy as np
import h5py
import tifffile

working_directory = r"/default/path"  # PERSONAL

source_path = os.path.join(working_directory, "uni_tp-0_ch-3_st-1-x00-y00-1-x00-y01_obj-bottom-bottom_cam-bottom_etc.lux.h5")
sink_path = os.path.join(working_directory, "uni_tp-0_ch-3_st-1-x00-y00-1-x00-y01_obj-bottom-bottom_cam-bottom_etc.tif")

with h5py.File(source_path, "r") as f:
    data = f["Data"]
    tifffile.imwrite(sink_path, data)
