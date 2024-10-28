import os

import tifffile
import numpy as np

# Load the 16-bit image
working_directory = r"E:\tto\test"
file_name = "result"
image_path = os.path.join(working_directory, fr"{file_name}.tif")
image_16bit = tifffile.imread(image_path)

# Normalize the 16-bit image to range [0, 1]
img_8bit = image_16bit - image_16bit.min()
img_8bit = img_8bit.astype("uint8")
img_8bit[img_8bit < 40] = 0
img_8bit[img_8bit > 100] = 100
img_8bit = (img_8bit - img_8bit.min()) / (img_8bit.max() - img_8bit.min())*255
img_8bit = img_8bit.astype("uint8")

# Define the current min and max pixel values (x and y)
# current_min = np.min(img_8bit)  # Replace with known x if needed
# current_max = np.max(img_8bit)  # Replace with known y if needed

# Normalize the image to range [0, 255]
# img_normalized = (img_8bit - current_min) / (current_max - current_min) * 255
# img_normalized = np.uint8(img_normalized)

tifffile.imwrite(os.path.join(working_directory, f"{file_name}_8b.tif"), img_8bit)
