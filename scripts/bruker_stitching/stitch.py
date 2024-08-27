import os
import json
import time

import stitching.stitching as st

import utils.utils as ut

raw_directory = r"/default/path"  # PERSONAL

parameters = {"study_params":
                  {"scanning_system": "",
                   "samples_to_process": [],
                   "channels_to_stitch": [5],
                   "re_process": True},
              "stitching": {"search_params": [5, 5, 5],
                            "z_subreg_alignment": [550, 600],
                            },
              }

sample_names = ut.get_sample_names(raw_directory, **parameters)

for sample in sample_names:
    sample_dir = os.path.join(raw_directory, sample)
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
        "overlap": ((2580/json_metadata["processingInformation"]["image_size_vx"]["width"])-1)/2,
    }
    # z = json_metadata["processingInformation"]["image_size_vx"]["depth"]
    # parameters["stitching"]["z_subreg_alignment"] = [int(z/2)-25, int(z/2)+25]
    with open(os.path.join(sample_dir, "scan_metadata.json"), 'w') as json_file:
        json.dump(scan_metadata, json_file, indent=4)

start_time = time.time()
st.stitch_samples(raw_directory, **parameters)
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
