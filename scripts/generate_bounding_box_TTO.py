import os

import numpy as np

working_dir = r"E:\tto\23-GUP030-0696-bruker\raw\ID039\xy5p0_z5p0\20240611-131019_Task_1_merged"

bounding_box = np.array([[227, 1300],  # Depth dim (in the stitched.tif file)
                         [100, 2248],  # Vertical dim
                         [30, 1644],  # Horizontal dim
                         ])

np.save(os.path.join(working_dir, "bounding_box.npy"), bounding_box)

merged_directory = r"E:\tto\23-GUP030-0696-bruker\raw\ID039\xy5p0_z5p0\20240611-131019_Task_1_merged"
bounding_box_path = os.path.join(merged_directory, "bounding_box.npy")
if os.path.exists(bounding_box_path):
    stitched_npy_data = np.load(os.path.join(merged_directory, "stitched_5.npy"))
    bounding_box = np.load(bounding_box_path)
    stitched_npy_clipped = stitched_npy_data[
                           bounding_box[0][0]: bounding_box[0][1],
                           bounding_box[1][0]: bounding_box[1][1],
                           bounding_box[2][0]: bounding_box[2][1]]
    np.save(os.path.join(merged_directory, "stitched_5.npy"), stitched_npy_clipped)