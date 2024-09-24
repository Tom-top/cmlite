import os

import numpy as np
import tifffile

working_dir = r"/default/path"  # PERSONAL
stitched_npy_file = os.path.join(working_dir, "stitched_3.npy")
tifffile.imwrite(os.path.join(working_dir, "stitched_3.tif"), np.load(stitched_npy_file))
