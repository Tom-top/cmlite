import os

import tifffile
import numpy as np

working_directory = r"/default/path"  # PERSONAL

image = tifffile.imread(os.path.join(working_directory, "stitched_3.tif"))
new_image = np.swapaxes(image, 0, 2)

tifffile.imwrite(os.path.join(working_directory, "stitched_3_flipped.tif"), new_image)
