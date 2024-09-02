import os

from natsort import natsorted
import numpy as np
import cv2


# Function to interpolate between two images
def interpolate_images(img1, img2, num_interpolations):
    interpolated_images = []
    for i in range(1, num_interpolations + 1):
        alpha = i / (num_interpolations + 1)
        interpolated_image = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        interpolated_images.append(interpolated_image)
    return interpolated_images


# Load images from folder
working_directory = r"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results\thick_slices"
categories = ["neurons", "Epen", "Vascular", "Oligo", "Astro", "OEC", "Immune"]
gene = "Hcrt"

for cat in categories:
    map = f"{gene}_{cat}"
    image_folder = os.path.join(working_directory, map)
    images = natsorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])
    frame_list = []

    # Number of interpolated frames between each pair of images
    num_interpolations = 0

    # Read and interpolate images
    for i in range(len(images) - 1):
        img1 = cv2.imread(os.path.join(image_folder, images[i]))
        img2 = cv2.imread(os.path.join(image_folder, images[i + 1]))

        # Append the original image
        frame_list.append(img1)

        # Append interpolated images
        interpolated_frames = interpolate_images(img1, img2, num_interpolations)
        frame_list.extend(interpolated_frames)

    # Append the last image
    frame_list.append(cv2.imread(os.path.join(image_folder, images[-1])))

    # Define video codec and create VideoWriter object
    height, width, layers = frame_list[0].shape
    video = cv2.VideoWriter(os.path.join(working_directory, f'{map}_video.avi'), cv2.VideoWriter_fourcc(*'XVID'),
                            3, (width, height))

    # Write frames to video
    for frame in frame_list:
        video.write(frame)

    # Release the VideoWriter object
    video.release()
