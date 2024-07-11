import os

import numpy as np
import cv2
import tifffile


def assign_random_colors(grayscale_image):
    # Find unique grayscale values in the image
    unique_values = np.unique(grayscale_image)

    # Create a dictionary to map each grayscale value to a random RGB color
    color_map = {}
    for value in unique_values:
        color_map[value] = np.random.randint(0, 256, size=3)

    # Create an empty image with the same dimensions as the grayscale image, but with 3 channels (RGB)
    colored_image = np.zeros((*grayscale_image.shape, 3), dtype=np.uint8)

    # Map each grayscale value to its corresponding RGB color
    for value, color in color_map.items():
        colored_image[grayscale_image == value] = color

    return colored_image


working_directory = r"U:\Users\TTO\phd_projects\lll\figures\microglial_density_whole-brain_baseline"
file_list = [os.path.join(working_directory, i) for i in os.listdir(working_directory) if i.split("_")[1] == "16b"]

for f in file_list:
    f_name = os.path.basename(f)
    new_f_name = f_name.split(".")[0] + "_rgb." + f_name.split(".")[-1]
    # Load the 16-bit grayscale image
    grayscale_image = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    # Ensure the image is 16-bit
    if grayscale_image.dtype != np.uint16:
        raise ValueError("The image is not 16-bit grayscale")
    # Assign random colors to the grayscale image
    colored_image = assign_random_colors(grayscale_image)
    tifffile.imwrite(os.path.join(working_directory, new_f_name), colored_image)