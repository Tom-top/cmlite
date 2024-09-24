import os

import numpy as np
import nibabel as nib
import tifffile

working_directory = r"/default/path"  # PERSONAL

source_path = os.path.join(working_directory, "result.nii.gz")
sink_path = os.path.join(working_directory, "result.tif")

img = nib.load(source_path)
# Get the image data as a numpy array
img_data = img.get_fdata()
tifffile.imwrite(sink_path, img_data)
