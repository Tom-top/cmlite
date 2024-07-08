import os

import numpy as np
import tifffile

working_dir = r""
stitched_npy_file = os.path.join(working_dir, "stitched_1.npy")
tifffile.imwrite(os.path.join(working_dir, "stitched_1.tif"), np.load(stitched_npy_file))