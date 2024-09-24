import os

import tifffile as tiff
import numpy as np

# Load the 3D TIFF file using tifffile
working_directory = r"/default/path"  # PERSONAL
input_tif = os.path.join(working_directory, 'heatmap_Hcrtr2_neurons_dynamic_bin.tif')  # Replace with the path to your 3D TIFF file
file_name = os.path.basename(input_tif).split(".")[0]
img_data = tiff.imread(input_tif)  # This will load the entire 3D image stack as a numpy array

# Normalize the image data to [0, 1]
img_data_normalized = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

# Convert the normalized data to uint8 (0-255 range)
img_data_uint8 = (img_data_normalized * 255).astype(np.uint8)

# Save the normalized 3D image as a new TIFF file
output_tif = os.path.join(working_directory, f'normalized_{file_name}.tif')
tiff.imwrite(output_tif, img_data_uint8, photometric='minisblack')

print("3D TIFF file has been normalized and saved as uint8.")
