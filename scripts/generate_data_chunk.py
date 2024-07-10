import os

import numpy as np
import tifffile

import IO.IO as io

working_directory = r"E:\tto\23-GUP030-0696-bruker\raw\ID014_an002992_g003_Brain_M3_rescan1\xy5p0_z5p0\2024-05-02_044404_merged"

x_s, y_s, z_s = 200, 200, 200
res_ratio = 2.5
delta = 600
io.generate_npy_chunk(
    os.path.join(working_directory, "stitched_5.npy"),
    np.array([[x_s * res_ratio, x_s * res_ratio + delta],
              [y_s * res_ratio, y_s * res_ratio + delta],
              [z_s * res_ratio, z_s * res_ratio + delta]]),
    os.path.join(working_directory, "chunk_stitched_5.npy"))
chunk = np.load(os.path.join(working_directory, "chunk_stitched_5.npy"))
tifffile.imwrite(os.path.join(working_directory, "chunk_stitched_5.tif"), chunk)