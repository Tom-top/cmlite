import os

import numpy as np
import h5py
import tifffile

raw_data_folder = r"E:\tto\23-GUP030-0696-bruker\raw\raw"

multistack = True
channels_to_stitch = [[3, 4]]
rows, columns = 7, 2
current_tile = 0

if multistack:
    for r in range(rows):
        for channels in channels_to_stitch:
            n_tiles_to_convert = len(channels) * rows * columns
            stitching_folder = os.path.join(os.path.dirname(raw_data_folder), f"processed_tiles_{channels[0]}")
            if not os.path.exists(stitching_folder):
                os.mkdir(stitching_folder)
            for n, channel in enumerate(channels):
                for c in range(columns):
                    current_tile += 1
                    c_idx = np.abs(1 - c)
                    if n == 0:
                        y = c
                    else:
                        y = c + 2
                    print(f"Reading and saving data for: channel{channel}_x0{r}_y0{c_idx} :"
                          f" stack_x0{r}_y0{y}. {current_tile}/{n_tiles_to_convert}")
                    path_to_stack = os.path.join(raw_data_folder,
                                                 f"stack_{n+1}-x0{r}-y0{c_idx}_channel_{channel}_obj_bottom")
                    path_to_h5 = os.path.join(path_to_stack, "Cam_bottom_00000.lux.h5")
                    f = h5py.File(path_to_h5, "r")
                    d = f["Data"]
                    data = d[()]
                    tifffile.imwrite(os.path.join(stitching_folder, f"stack_[{r} x {y}]_{channel}.tif"), data)
else:
    for channels in channels_to_stitch:
        for n, channel in enumerate(channels):
            stitching_folder = os.path.join(os.path.dirname(raw_data_folder), f"processed_tiles_{channel}")
            if not os.path.exists(stitching_folder):
                os.mkdir(stitching_folder)
            channel_path = os.path.join(stitching_folder, f"channel_{channel}")
            if not os.path.exists(channel_path):
                os.mkdir(channel_path)
            for r in range(rows):
                for c in range(columns):
                    path_to_stack = os.path.join(raw_data_folder,
                                                 f"stack_1-x0{r}-y0{c}_channel_{channel}_obj_bottom")
                    path_to_h5 = os.path.join(path_to_stack, "Cam_bottom_00000.lux.h5")
                    f = h5py.File(path_to_h5, "r")
                    d = f["Data"]
                    data = d[()]
                    tifffile.imwrite(os.path.join(stitching_folder, f"stack_[{r} x {c}]_{channel}.tif"), data)