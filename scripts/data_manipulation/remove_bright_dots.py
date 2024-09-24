import os
import numpy as np
import cv2
import tifffile

# Paths and data loading
working_directory = r"E:\tto\24-VOILE-0726\ID005_an000005_g002_Brain_M4"
image_path = os.path.join(working_directory,
                          "uni_tp-0_ch-3_st-1-x00-y00-1-x00-y01_obj-bottom-bottom_cam-bottom_etc.tif")
save_intermediate_result = False

print("Loading image data...")
image_data = tifffile.imread(image_path)

print("Cropping image data (if necessary)...")
image_data_crop = image_data
if save_intermediate_result:
    tifffile.imwrite(os.path.join(working_directory, "crop.tif"), image_data_crop)

print("Applying thresholding...")
threshold_value = 1600  # Lower threshold to include more of the bright area (default: 1800)
binary_image = np.zeros_like(image_data_crop, dtype=np.uint8)
for i in range(image_data_crop.shape[0]):
    _, binary_image[i] = cv2.threshold(image_data_crop[i], threshold_value, 255, cv2.THRESH_BINARY)
if save_intermediate_result:
    tifffile.imwrite(os.path.join(working_directory, "binary.tif"), binary_image)

print("Performing morphological operations...")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # (default: 5, 5)
dilated_image = np.zeros_like(binary_image)
for i in range(image_data_crop.shape[0]):
    binary_slice = np.squeeze(binary_image[i])
    dilated_slice = cv2.dilate(binary_slice, kernel, iterations=1)

    # Connected Component Analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated_slice, connectivity=8)
    large_component_threshold = 1000  # Threshold for large components (adjust as needed)

    # Filter out large components
    for label in range(1, num_labels):  # Start from 1 to skip the background
        if stats[label, cv2.CC_STAT_AREA] > large_component_threshold:
            dilated_slice[labels == label] = 0

    dilated_image[i] = np.reshape(dilated_slice, binary_image[i].shape)

if save_intermediate_result:
    tifffile.imwrite(os.path.join(working_directory, "dilated_filtered.tif"), dilated_image)

# Inpainting using OpenCV
print("Starting inpainting...")
inpainted_image = np.zeros_like(image_data_crop, dtype=np.uint16)

for i in range(image_data_crop.shape[0]):
    print(f"Inpainting slice {i + 1}/{image_data_crop.shape[0]}...")
    inpainted_slice = cv2.inpaint(image_data_crop[i], dilated_image[i].astype(np.uint8), inpaintRadius=3,
                                  flags=cv2.INPAINT_TELEA)
    inpainted_image[i] = inpainted_slice

# Saving the final inpainted image
print("Saving the final inpainted image...")
tifffile.imwrite(os.path.join(working_directory, "clean_16bit_inpainted.tif"), inpainted_image)

print("Inpainting complete!")
