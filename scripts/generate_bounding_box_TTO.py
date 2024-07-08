import os

import numpy as np

working_dir = r"E:\tto\23-GUP030-0696-bruker\raw\ID035_an161012_g006_Brain_M3\xy5p0_z5p0\2024-01-10_001628_merged"

bounding_box = np.array([[110, 1314],  # Depth dim (in the stitched.tif file)
                         [236, 2828],  # Vertical dim
                         [276, 2152],  # Horizontal dim
                         ])

np.save(os.path.join(working_dir, "bounding_box.npy"), bounding_box)