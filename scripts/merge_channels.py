import os

import tifffile

working_directory = r"d"

stack_1_path = os.path.join(working_directory, "stitched_3_5um.tif")
stack_2_path = os.path.join(working_directory, "stitched_4_5um.tif")

stack_1 = tifffile.imread(stack_1_path)
stack_2 = tifffile.imread(stack_2_path)
merge = stack_1.copy()
merge[835:] = stack_2[835:]
tifffile.imwrite(os.path.join(working_directory, "stitched_3.tif"), merge)