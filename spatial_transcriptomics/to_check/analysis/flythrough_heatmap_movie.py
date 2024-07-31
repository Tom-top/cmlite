import os

import cv2
import numpy as np
import tifffile

def read_tiff_stack(file_path):
    """Read a TIFF stack and return it as a numpy array."""
    with tifffile.TiffFile(file_path) as tif:
        images = tif.asarray()
    return images

def make_movie(tiff_files, output_file, frame_size, fps=10):
    """Create a movie from TIFF stacks."""
    # Open the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    # Read the stacks and assume they are of the same shape
    stacks = [read_tiff_stack(file) for file in tiff_files]
    n_frames = stacks[0].shape[0]

    # Create each frame of the video
    for i in range(n_frames):
        frames = [stack[i] for stack in stacks]
        combined_frame = np.concatenate(frames, axis=1)  # Combine frames horizontally
        combined_frame = cv2.normalize(combined_frame, None, 0, 255, cv2.NORM_MINMAX)  # Normalize
        combined_frame = np.uint8(combined_frame)  # Convert to uint8

        # If the images are grayscale, convert them to a format OpenCV can write
        if len(combined_frame.shape) == 2:
            combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_GRAY2BGR)

        video_writer.write(combined_frame)

    video_writer.release()

# Example usage
atlas_with_outlines = r"E:\tto\results\cell_positions\general\gubra_v6_atlas_o.tif"
atlas_outlines = r"E:\tto\results\cell_positions\general\gubra_v6_atlas_outlines.tif"
heatmap_directory = r"E:\tto\results\heatmaps\Zhuang-ABCA-1"
genes_of_interest = ["Bdnf"]
tiff_files = [os.path.join(heatmap_directory, fr"{g}\{g}_heatmap.tif") for g in genes_of_interest]  # Add your TIFF file paths here

movie_directory = r"E:\tto\results\heatmaps\movies"
output_file = os.path.join(movie_directory, f'{"_".join(genes_of_interest)}_movie.mp4')

# Assume all tiff files have the same dimensions, read first to get frame size
example_stack = read_tiff_stack(tiff_files[0])
frame_height, frame_width = example_stack.shape[1:3]
frame_size = (frame_width * len(tiff_files), frame_height)  # width, height

make_movie(tiff_files, output_file, frame_size)
