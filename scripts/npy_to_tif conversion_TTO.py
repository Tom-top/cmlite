import os

import numpy as np
import tifffile

working_dir = r"E:\tto\23-GUP030-0696-bruker\raw\ID035_an161012_g006_Brain_M3\xy5p0_z5p0\2024-01-10_001628_merged"
stitched_npy_file = os.path.join(working_dir, "stitched_1.npy")
tifffile.imwrite(os.path.join(working_dir, "stitched_1.tif"), np.load(stitched_npy_file))