import os

import tifffile
from imaris_ims_file_reader.ims import ims

working_directory = r"E:\tto\24-VOILE-0726\ID004_an000004_g002_Brain_M3"
imaris_file_path = os.path.join(working_directory, "original.ims")

imaris_file = ims(imaris_file_path)
imaris_data = imaris_file[0, 0]

print(imaris_data.shape)

tifffile.imwrite(os.path.join(working_directory, "uni_tp-0_ch-3_st-1-x00-y00-1-x00-y01_obj-bottom-bottom_cam-bottom_etc.tif"), imaris_data)
