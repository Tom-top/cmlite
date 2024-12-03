import os

from natsort import natsorted
import numpy as np
import tifffile

# Replace this import with the actual location of the 'generate_npy_chunk' function
import IO.IO as io
import utils.utils as ut

# Define the working directory and other parameters
working_directory = (r"/mnt/data/Grace/projectome/1_earmark_gfp_20tiles_2z_3p25zoom_singleill/raw/sample_1")
img_path = os.path.join(working_directory, "stitched_0.npy")
img_shape, _ = io.get_npy_metadata(img_path)
img_shape = img_shape[::-1]

starts = np.arange(0, img_shape[1], 1000)
res_ratio = 1
coords = [all, 0, all]
deltas = [all, 1000, all]

for n, coordinates in enumerate(coords):
    if coordinates == all:
        coords[n] = 0
        deltas[n] = img_shape[n]

for m, s in enumerate(starts):

    saving_dir = ut.create_dir(os.path.join(working_directory, f"chunk_{m}"))

    start = s * res_ratio
    end = s * res_ratio + deltas[1]
    if end > img_shape[1]:
        end = img_shape[1]

    # Call the function to generate the chunk with memory mapping enabled
    io.generate_chunk(
        img_path,
        np.array([[coords[0] * res_ratio, coords[0] * res_ratio + deltas[0]],
                  [start, end],
                  [coords[2] * res_ratio, coords[2] * res_ratio + deltas[2]]]),
        saving_dir,
        memmap=True
    )
