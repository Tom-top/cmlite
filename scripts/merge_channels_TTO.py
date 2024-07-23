import os

import tifffile

working_directory = r"E:\tto\23-GUP030-0696-bruker\raw\ID039\xy5p0_z5p0\20240611-131019_Task_1_merged\channel_merging"

stack_1_path = os.path.join(working_directory, "stitched_5_5um.tif")
stack_2_path = os.path.join(working_directory, "stitched_6_5um.tif")

stack_1 = tifffile.imread(stack_1_path)
stack_2 = tifffile.imread(stack_2_path)
merge = stack_1.copy()
merge[835:] = stack_2[835:]
tifffile.imwrite(os.path.join(working_directory, "stitched_5.tif"), merge)

# import numpy as np
#
# merged_data = tifffile.imread(os.path.join(working_directory, "stitched_5.tif"))
# np.save(os.path.join(working_directory, "stitched_5.npy"), np.swapaxes(merged_data, 0, 2))