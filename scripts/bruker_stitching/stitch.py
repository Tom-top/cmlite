import os
import json
import time

import tifffile

import stitching.stitching as st

import utils.utils as ut

raw_directory = r"/default/path"  # PERSONAL
resampled = True

parameters = {"study_params":
                  {"scanning_system": "",
                   "samples_to_process": [],
                   "channels_to_stitch": [5],
                   "re_process": True},
              "stitching": {"search_params": [10, 10, 10],
                            },
              }

sample_names = ut.get_sample_names(raw_directory, **parameters)

for sample in sample_names:
    sample_dir = os.path.join(raw_directory, sample)
    if resampled:
        processed_folder = [os.path.join(sample_dir, i) for i in os.listdir(sample_dir)
                            if os.path.isdir(os.path.join(sample_dir, i)) and i.startswith("processed")]
        if len(processed_folder) > 0:
            processed_folder = processed_folder[0]
            tiff_stack = [os.path.join(processed_folder, i) for i in os.listdir(processed_folder)
                          if i.endswith(".tif")][0]
            stack = tifffile.imread(tiff_stack)
            stack_shape = stack.shape
            scan_metadata = {
                "tile_x": stack_shape[2],
                "tile_y": stack_shape[1],
                "tile_z": stack_shape[0],  # This is not a mistake, Z comes first in the tif shape
                "x_res": 5,
                "y_res": 5,
                "z_res": 10,
                "overlap": 0.1298828125,
            }
            mid_z_plane = int(stack_shape[1]/2)
            parameters["stitching"]["z_subreg_alignment"] = [mid_z_plane, mid_z_plane + 50]
            with open(os.path.join(sample_dir, "scan_metadata.json"), 'w') as json_file:
                json.dump(scan_metadata, json_file, indent=4)
    else:
        timestamp_folder = [os.path.join(sample_dir, i) for i in os.listdir(sample_dir)
                            if os.path.isdir(os.path.join(sample_dir, i)) and not i.startswith("processed")][0]
        raw_dir = os.path.join(timestamp_folder, "raw")
        stack_dir = os.path.join(raw_dir, os.listdir(raw_dir)[0])
        file_name = os.path.join(stack_dir, "Cam_bottom_00000.json")
        json_metadata = ut.load_json_file(file_name)
        scan_metadata = {
            "tile_x": json_metadata["processingInformation"]["image_size_vx"]["width"],
            "tile_y": json_metadata["processingInformation"]["image_size_vx"]["height"],
            "tile_z": json_metadata["processingInformation"]["image_size_vx"]["depth"],
            "x_res": json_metadata["processingInformation"]["voxel_size_um"]["width"],
            "y_res": json_metadata["processingInformation"]["voxel_size_um"]["height"],
            "z_res": json_metadata["processingInformation"]["voxel_size_um"]["depth"],
            # "overlap": ((2580/json_metadata["processingInformation"]["image_size_vx"]["width"])-1)/2,,
            "overlap": 0.1,
        }
        mid_z_plane = int(json_metadata["processingInformation"]["image_size_vx"]["depth"] / 2)
        parameters["stitching"]["z_subreg_alignment"] = [mid_z_plane, mid_z_plane + 50]
        with open(os.path.join(sample_dir, "scan_metadata.json"), 'w') as json_file:
            json.dump(scan_metadata, json_file, indent=4)

start_time = time.time()
st.stitch_samples(raw_directory, **parameters)
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
