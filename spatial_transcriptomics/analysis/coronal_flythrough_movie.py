import cv2
import os
import glob
from natsort import natsorted

def create_flythrough(folder_path, category, output_file, frame_size, fps):
    # Get all png images in the folder
    img_array = []
    for filename in natsorted(glob.glob(f'{folder_path}/{category}_*.png')):
        img = cv2.imread(filename)
        img = cv2.resize(img, frame_size)
        img_array.append(img)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    # Write each frame to the video
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

# Example usage
category = "cluster"
folder_path = r'/mnt/data/spatial_transcriptomics/results/mpd5_pick1/Pick-1_vs_Vehicle/transcriptomics/slices'  # Replace with your folder path
output_file = os.path.join(os.path.dirname(folder_path), f"coronal_flythrough_{category}.mp4")  # Output filename
frame_size = (1920, 1440)  # Frame size, change as needed
fps = 30  # Frames per second

create_flythrough(folder_path, category, output_file, frame_size, 3)