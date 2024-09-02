import os
import re
import time

import numpy as np
import h5py
import tifffile

import IO.file_utils as fu

import resampling.resampling as res

import utils.utils as ut

raw_data_folder = r"/default/path"  # PERSONAL

start_time = time.time()
multistack = True
downsample = True
channels_to_stitch = [[3, 4]]


def find_max_indices(folder_names):
    max_x = 0
    max_y = 0

    # Define a regular expression to extract x and y values
    pattern = re.compile(r'-x(\d+)-y(\d+)_')

    for folder in folder_names:
        match = pattern.search(folder)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    return max_x + 1, max_y + 1


rows, columns = find_max_indices(os.listdir(raw_data_folder))
current_tile = 0

if multistack:
    timestamp_folder = os.path.dirname(raw_data_folder)
    sample_folder = os.path.dirname(timestamp_folder)
    for r in range(rows):
        for channels in channels_to_stitch:
            n_tiles_to_convert = len(channels) * rows * columns
            stitching_folder = os.path.join(sample_folder, f"processed_tiles_{channels[0]}")
            if not os.path.exists(stitching_folder):
                os.mkdir(stitching_folder)
            for n, channel in enumerate(channels):
                for c in range(columns):
                    current_tile += 1
                    c_idx = np.abs((columns - 1) - c)
                    if n == 0:
                        y = c
                    else:
                        y = c + columns
                    # print(n+1, r, c_idx, r, y)
                    print(f"Reading and saving data for: channel{channel}_x0{r}_y0{c_idx} :"
                          f" stack_x0{r}_y0{y}. {current_tile}/{n_tiles_to_convert}")
                    path_to_stack = os.path.join(raw_data_folder,
                                                 f"stack_{n+1}-x0{r}-y0{c_idx}_channel_{channel}_obj_bottom")
                    path_to_h5 = os.path.join(path_to_stack, "Cam_bottom_00000.lux.h5")
                    f = h5py.File(path_to_h5, "r")
                    d = f["Data"]
                    data = d[()]
                    path_to_tif = os.path.join(stitching_folder, f"stack_[{r} x {y}]_{channels[0]}.tif")
                    if downsample:
                        path_to_json = os.path.join(path_to_stack, "Cam_bottom_00000.json")
                        tile_metadata = ut.load_json_file(path_to_json)
                        voxel_size = tile_metadata["processingInformation"]["voxel_size_um"]
                        x_vs, y_vs, z_vs = voxel_size["width"], voxel_size["height"], voxel_size["depth"]
                        resample_parameter = {
                            "source_resolution": (x_vs, y_vs, z_vs),
                            "sink_resolution": (5, 5, 5),
                            "processes": None,
                            "verbose": True,
                            "method": "memmap",
                        }
                        fu.delete_file(path_to_tif)
                        res.resample(np.swapaxes(data, 0, 2), sink=path_to_tif, **resample_parameter)
                    else:
                        tifffile.imwrite(path_to_tif, data)
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

end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
