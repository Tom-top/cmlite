import os

import numpy as np
import tifffile

# Load the TIFF stacks
working_directore = r"E:\tto\23-GUP030-0696\raw\ID888_an000888_g010_brain_M3\xy5p0_z5p0\2024-08-29_194534_merged\shape_detection_350"
stack1 = tifffile.imread(os.path.join(working_directore, "stitched_3.tif"))
stack1 = np.swapaxes(stack1, 0, 2)
stack1 = stack1.astype(np.uint16)  # or use the appropriate data type

stack2 = tifffile.imread(os.path.join(working_directore, "shape_detection_filtered.tif"))
stack2 = stack2.astype(np.uint16)

# Ensure the stacks have the same dimensions
assert stack1.shape == stack2.shape, "Image stacks must have the same dimensions!"

# Combine them into a multi-channel image
combined = np.stack((stack1, stack2), axis=-1)

# Save the combined stack as a multi-channel TIFF
tifffile.imwrite(os.path.join(working_directore, "combined.tif"), combined, photometric='minisblack', metadata=None)
