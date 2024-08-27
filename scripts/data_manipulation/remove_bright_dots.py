import os
import numpy as np
import cv2
import h5py
import tifffile

# Paths and data loading
working_directory = r"E:\tto\ID001_an000001_g001_Brain_M3_rescan1\xy5p0_z10p0\2024-07-31_045420_merged"
image_path = os.path.join(working_directory, "uni_tp-0_ch-5_st-1-x00-y00-1-x00-y01_obj-bottom-bottom_cam-bottom_etc.lux.h5")
save_intermediate_result = False

with h5py.File(image_path, 'r') as h5_file:
    image_data = np.array(h5_file["Data"])

# Cropping the image
# xs, ys, zs = 1900, 1900, 300
# delta = 500
# image_data_crop = image_data[zs:zs+delta, xs:xs+delta, ys:ys+delta]
image_data_crop = image_data
if save_intermediate_result:
    tifffile.imwrite(os.path.join(working_directory, "crop.tif"), image_data_crop)

# Thresholding with a potentially lower value to capture more of the bright spots
threshold_value = 1800  # Lower threshold to include more of the bright area
binary_image = np.zeros_like(image_data_crop, dtype=np.uint8)
# Apply thresholding slice by slice along the depth dimension
for i in range(image_data_crop.shape[0]):  # Iterate over the depth slices
    _, binary_image[i] = cv2.threshold(image_data_crop[i], threshold_value, 255, cv2.THRESH_BINARY)
if save_intermediate_result:
    tifffile.imwrite(os.path.join(working_directory, "binary.tif"), binary_image)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Further increased kernel size
dilated_image = np.zeros_like(binary_image)

# Process each slice in 3D data
for i in range(image_data_crop.shape[0]):  # Iterate through slices
    # Ensure the slice is treated as 2D
    binary_slice = np.squeeze(binary_image[i])  # Remove singleton dimensions if present
    dilated_slice = cv2.dilate(binary_slice, kernel, iterations=1)  # Dilation to expand mask
    # Ensure the output is correctly reshaped if necessary
    dilated_image[i] = np.reshape(dilated_slice, binary_image[i].shape)
if save_intermediate_result:
    tifffile.imwrite(os.path.join(working_directory, "dilated.tif"), dilated_image)

# Create the cleaned image by replacing the bright areas
cleaned_image = image_data_crop.copy()

for i in range(cleaned_image.shape[0]):  # Apply replacement slice by slice
    cleaned_image[i][dilated_image[i] == 255] = 800  # Replace bright spots with a fixed value (800)
#
# # Inpainting slice by slice
# inpainted_image = np.zeros_like(cleaned_image)
#
# for i in range(cleaned_image.shape[0]):
#     # Convert to 8-bit single-channel
#     cleaned_image_8bit = cv2.convertScaleAbs(cleaned_image[i], alpha=(255.0 / np.max(cleaned_image[i])))
#
#     # Ensure dilated_image is in the correct format (8-bit 1-channel binary mask)
#     dilated_image_8bit = dilated_image[i].astype(np.uint8)
#     dilated_image_8bit[dilated_image_8bit > 0] = 255  # Ensure it's binary (0 or 255)
#
#     # Apply inpainting on each slice
#     inpainted_image[i] = cv2.inpaint(cleaned_image_8bit, dilated_image_8bit, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Save the inpainted result
tifffile.imwrite(os.path.join(working_directory, "clean.tif"), cleaned_image)
