import numpy as np
import cv2
import os


def interpolate_images(image1, image2, num_steps):
    """Linearly interpolate between two images."""
    interpolated_images = []
    for t in range(num_steps):
        alpha = t / float(num_steps - 1)
        interpolated = (1 - alpha) * image1 + alpha * image2
        interpolated_images.append(np.uint8(interpolated))  # Ensure data type is uint8 for images
    return interpolated_images


def create_video_from_images(image_list, output_video_path, duration=30, fps=30):
    """Create a smooth video by interpolating between images."""
    num_images = len(image_list)
    total_frames = duration * fps  # Total number of frames in the video
    frames_per_image = total_frames // (num_images - 1)  # Number of frames to interpolate between each image

    height, width, _ = image_list[0].shape  # Assuming all images have the same size
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i in range(num_images - 1):
        # Interpolate between the current image and the next one
        interpolated_frames = interpolate_images(image_list[i], image_list[i + 1], frames_per_image)
        for frame in interpolated_frames:
            video_writer.write(frame)

    video_writer.release()


working_directory = r"E:\tto\23-GUM041-0522"

# Assuming you have n images stored in a list of numpy arrays called `image_list`
image_list = [cv2.imread(os.path.join(working_directory, f'W{i}_backward.tif')) for i in [1, 4, 8, 12, 16, 26]]
output_video_path = os.path.join(working_directory, 'asyn_spread_video.mp4')

# Create a video with about 30 seconds duration
create_video_from_images(image_list, output_video_path, duration=10, fps=10)
