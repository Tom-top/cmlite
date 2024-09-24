import os

import tifffile
import numpy as np

# Load the 64-bit image
working_directory = r"E:\tto\23-GUP030-0696\raw\ID888_an000888_g010_brain_M3\xy5p0_z5p0\2024-08-29_194534_merged\shape_detection_350"
file_name = "background_removal"
image_path = os.path.join(working_directory, fr"{file_name}.tif")
image_64bit = tifffile.imread(image_path)

# Check the image data type
print(f"Original image dtype: {image_64bit.dtype}")

# Normalize or scale image data to 16-bit range
if image_64bit.dtype == np.float64:
    # For float64 images, normalize to 0-65535
    image_16bit = np.clip((image_64bit - np.min(image_64bit)) / (np.max(image_64bit) - np.min(image_64bit)) * 65535, 0, 65535).astype(np.uint16)
elif image_64bit.dtype == np.uint64:
    # For uint64 images, clip values to 0-65535
    image_16bit = np.clip(image_64bit, 0, 65535).astype(np.uint16)
else:
    raise TypeError("Unsupported image data type. Only float64 and uint64 are supported.")

# Save the 16-bit image
output_path = os.path.join(working_directory, fr"{file_name}_16b.tif")
tifffile.imwrite(output_path, image_16bit)

print(f"Converted image saved to {output_path} with dtype {image_16bit.dtype}.")
