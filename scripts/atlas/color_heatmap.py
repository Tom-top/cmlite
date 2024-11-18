"""
This script takes as an input a map aligned/transformed to the Gubra v6 atlas and applies ABA colors to it.
"""

import os

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut
import spatial_transcriptomics.old.utils.utils as sut

ATLAS_USED = "gubra"
ANO_DIRECTORY = r"resources\atlas"
REFERENCE_FILE = os.path.join(ANO_DIRECTORY, fr"{ATLAS_USED}_reference_mouse.tif")
REFERENCE = tifffile.imread(REFERENCE_FILE)

working_directory = r"E:\tto\test"
# map_names = ["W1", "W4", "W8", "W12", "W16", "W26"]
map_names = ["ano"]
overlay_template = False

for map_name in map_names:
    map_directory = os.path.join(working_directory, map_name)
    # Load the 16-bit grayscale image
    # grayscale_image = tifffile.imread(mask_path).astype("uint16")
    grayscale_image = tifffile.imread(os.path.join(map_directory, "gubra_annotation_mouse.tif"))
    # grayscale_image = np.array([tifffile.imread(os.path.join(map_directory, i))
    #                             for i in natsorted(os.listdir(map_directory))])
    grayscale_image = np.flip(np.swapaxes(grayscale_image, 0, 1), 1)
    # Normalize the grayscale image to range [0, 1] for alpha mapping
    # clip_max = 50
    clip_max = 0
    # grayscale_image[grayscale_image > clip_max] = clip_max
    grayscale_image[grayscale_image > clip_max] = 255
    # grayscale_norm = grayscale_image / clip_max
    grayscale_norm = grayscale_image / 255
    # colored_image = np.stack((grayscale_image,)*3, axis=-1)
    colored_image = np.stack((grayscale_image,)*4, axis=-1)

    ########################################################################################################################
    # GENERATE RGB IMAGE OF THE MASK
    ########################################################################################################################

    ANNOTATION_FILE = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.tif")
    ANO = np.transpose(tifffile.imread(ANNOTATION_FILE), (1, 2, 0))
    ANO_JSON = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.json")
    metadata = ut.load_json_file(ANO_JSON)
    unique_ids = np.unique(ANO)
    n_unique_ids = len(unique_ids)

    aba_colored_mask = colored_image.copy() / 255

    for n, uid in enumerate(unique_ids):
        if uid not in [0, 5000]:
            uid_color = sut.find_dict_by_key_value(metadata, uid)["color_hex_triplet"]
            uid_name = sut.find_dict_by_key_value(metadata, uid)["acronym"]
            if uid_color is not None:
                ut.print_c(f"[INFO] Coloring mask for region: {uid_name}; {n+1}/{n_unique_ids}")
                # uid_color_hex = np.array([i * 255 for i in ut.hex_to_rgb("#" + uid_color)])
                uid_color_hex = ut.hex_to_rgb("#" + uid_color)
                uid_mask = ANO == uid
                mask_intersections = np.logical_and(uid_mask, grayscale_image)
                # aba_colored_mask[mask_intersections] = uid_color_hex

                voxel_indices = np.argwhere(mask_intersections)
                for idx in voxel_indices:
                    z, y, x = idx  # Extract the coordinates
                    aba_colored_mask[z, y, x] = list(uid_color_hex) + [grayscale_norm[z, y, x]]

    def project_first_nonzero(aba_colored_mask, axis, direction="forward"):
        # Move the chosen axis to the front
        aba_colored_mask = np.moveaxis(aba_colored_mask, axis, 0)
        depth, height, width, channels = aba_colored_mask.shape

        first_nonzero_image = np.zeros((height, width, channels), dtype=aba_colored_mask.dtype)

        # Determine the range to iterate over based on the direction
        if direction == "forward":
            range_iter = range(depth)  # Project from up to down or front to back
        elif direction == "backward":
            range_iter = range(depth - 1, -1, -1)  # Project from down to up or back to front

        # Iterate over the chosen axis in the specified direction
        for i in range_iter:
            mask = np.any(first_nonzero_image == 0, axis=2) & np.any(aba_colored_mask[i] > 0, axis=2)
            first_nonzero_image[mask] = aba_colored_mask[i][mask]

        return first_nonzero_image


    def max_projection_with_alpha(image, axis=0):
        """
        Perform a max projection along the specified axis, keeping values with the highest alpha channel.

        Args:
        - image: A 4D numpy array of shape (D, H, W, 4), where the last dimension represents RGB + Alpha.
        - axis: The axis along which to project (0, 1, or 2).

        Returns:
        - max_projection: The resulting 3D array (after projection) with the highest alpha values kept.
        """

        # Extract the alpha channel (4th channel in the last axis)
        alpha_channel = image[..., 3]  # Shape is (D, H, W) depending on the input shape

        # Get the indices of the maximum alpha values along the specified axis
        max_alpha_indices = np.argmax(alpha_channel, axis=axis)  # Shape of (H, W) or (D, W) or (D, H)

        # Create an array to store the max projection result with the same spatial dimensions (height, width)
        max_projection_shape = list(image.shape)
        max_projection_shape[axis] = 1  # Collapse the chosen axis
        max_projection_shape = tuple(
            [max_projection_shape[dim] for dim in range(len(max_projection_shape)) if dim != axis])  # Remove collapsed axis

        # Initialize the max projection array
        max_projection = np.zeros(max_projection_shape[:-1] + (4,), dtype=image.dtype)

        # Now, we need to use np.take_along_axis for each channel (R, G, B, Alpha)
        for i in range(4):  # For each of the RGB and alpha channels
            # Use np.take_along_axis to select values along the given axis
            taken_along_axis = np.take_along_axis(image[..., i], np.expand_dims(max_alpha_indices, axis=axis), axis=axis)

            # Place the result in the appropriate position in the max_projection array
            if axis == 0:
                max_projection[:, :, i] = taken_along_axis[0, :, :]
            elif axis == 1:
                max_projection[:, :, i] = taken_along_axis[:, 0, :]
            elif axis == 2:
                max_projection[:, :, i] = taken_along_axis[:, :, 0]

        return max_projection


    ori = "horizontal"
    orix, oriy = 2, 0
    xlim, ylim = 369, 512

    # Choose the axis over which to project (0 for depth, 1 for height, 2 for width)
    axis = 1  # Example: choose depth
    direction = "forward"  # Choose "forward" or "backward" for projection direction

    # Perform a max projection along axis 1 (e.g., along the height dimension)
    # aba_colored_mask_8b = aba_colored_mask*255
    # aba_colored_mask_8b = aba_colored_mask_8b.astype("uint8")
    # tifffile.imwrite(os.path.join(working_directory, "test.tif"), aba_colored_mask_8b)
    first_nonzero_image = max_projection_with_alpha(aba_colored_mask, axis=1)
    # first_nonzero_image = project_first_nonzero(colored_image, axis, direction)

    # Create an RGBA version of the first_nonzero_image
    # colored_image_rgba = np.zeros((*first_nonzero_image.shape[:2], 4), dtype=np.float32)
    # colored_image_rgba[..., :4] = first_nonzero_image
    # colored_image_rgba[..., 3] = np.where(np.any(first_nonzero_image > 0, axis=2), 0.5, 0.0)
    # first_nonzero_image[..., 3] = np.where(np.any(first_nonzero_image > 0, axis=2), 1, 0.0)

    # Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if overlay_template:
        ax.imshow(np.rot90(np.max(REFERENCE, axis=orix))[::-1], cmap='gray_r', alpha=0.3)
    ax.imshow(first_nonzero_image)
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.invert_yaxis()
    ax.axis('off')

    # Save the figure
    fig.savefig(os.path.join(working_directory, f"{map_name}_{direction}_aba.tif"), dpi=600)
