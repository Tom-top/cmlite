import os

import numpy as np
import cv2
import tifffile

from utils.utils import assign_random_colors


working_directory = r"/default/path"  # PERSONAL
file_list = []
for f in os.listdir(working_directory):
    f_split = f.split("_")
    if len(f_split) > 1:
        if f_split[1] in ["16b", "16b.tif"] and not f.endswith("rgb.tif"):
            file_list.append(os.path.join(working_directory, f))

for f in file_list:
    f_name = os.path.basename(f)
    new_f_name = f_name.split(".")[0] + "_rgb." + f_name.split(".")[-1]
    # Load the 16-bit grayscale image
    grayscale_image = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    # Ensure the image is 16-bit
    if grayscale_image.dtype != np.uint16:
        raise ValueError("The image is not 16-bit grayscale")
    # Assign random colors to the grayscale image
    colored_image = assign_random_colors(grayscale_image)
    tifffile.imwrite(os.path.join(working_directory, new_f_name), colored_image)
