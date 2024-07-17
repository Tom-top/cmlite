import os

import numpy as np
import tifffile

working_dir = r"E:\tto\23-GUP030-0696-bruker\raw\ID039\xy5p0_z5p0\20240611-131019_Task_1_merged"
stitched_npy_file = os.path.join(working_dir, "stitched_3.npy")
tifffile.imwrite(os.path.join(working_dir, "stitched_3.tif"), np.load(stitched_npy_file))