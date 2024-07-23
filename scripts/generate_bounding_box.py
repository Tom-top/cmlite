import os

import numpy as np

working_dir = r""  # PERSONAL

bounding_box = np.array([[110, 1314],  # Depth dim (in the stitched.tif file)
                         [236, 2828],  # Vertical dim
                         [276, 2152],  # Horizontal dim
                         ])

np.save(os.path.join(working_dir, "bounding_box.npy"), bounding_box)
