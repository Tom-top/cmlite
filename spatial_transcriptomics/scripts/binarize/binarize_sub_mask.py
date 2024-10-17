import os
import numpy as np
import tifffile

# Define paths and cutoff value for binarization
zscore_maps_directory = r"/default/path"  # PERSONAL
zscore_map_name = "GUS2022-189-LY"
zscore_map_directory = os.path.join(zscore_maps_directory, zscore_map_name)

# Load the whole brain mask
whole_brain_mask = tifffile.imread(os.path.join(zscore_map_directory, "hemisphere_mask.tif"))

# Define the range for the region you want to preserve
range = np.array([[327, 380], [100, 150], [169, 203]])

# Create a new mask where everything is set to 255 initially
masked_brain = np.full(whole_brain_mask.shape, 0, dtype=whole_brain_mask.dtype)

# Now copy the region defined by 'range' from the original mask to the new mask
masked_brain[range[0, 0]:range[0, 1],
             range[1, 0]:range[1, 1],
             range[2, 0]:range[2, 1]] = whole_brain_mask[range[0, 0]:range[0, 1],
                                                        range[1, 0]:range[1, 1],
                                                        range[2, 0]:range[2, 1]]

# Save the masked image as a new file
tifffile.imwrite(os.path.join(zscore_map_directory, "peri-pag_mask.tif"), masked_brain)
