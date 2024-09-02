import os

import tifffile
from imaris_ims_file_reader.ims import ims

working_directory = r"E:\tto\24-VOILE-0726\ID003_an000003_g001_Brain_M3"
imaris_file_path = os.path.join(working_directory, "Z0000.ims")

imaris_file = ims(imaris_file_path)
imaris_data = imaris_file[0, 0]

print(imaris_data.shape)

tifffile.imwrite(os.path.join(working_directory, "Z0000.tif"), imaris_data)
