import os

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation

def dilate_3d_mask(mask, structure=None, iterations=1):
    """
    Dilates a 3D binary mask.

    Parameters:
    - mask (numpy.ndarray): A 3D binary array to be dilated.
    - structure (numpy.ndarray, optional): Structuring element for the dilation.
      If None, a cubic structuring element is used.
    - iterations (int, optional): Number of dilation iterations. Default is 1.

    Returns:
    - numpy.ndarray: A 3D binary array after dilation.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 3:
        raise ValueError("Input mask must be a 3D numpy array.")
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)  # Default cubic structuring element
    return binary_dilation(mask, structure=structure, iterations=iterations)

working_directory = "/mnt/data/Thomas/PPN_CUN"

mask = tifffile.imread(os.path.join(working_directory, "whole_brain_mask.tif"))
dilated_mask = dilate_3d_mask(mask, iterations=3) * 255
tifffile.imwrite(os.path.join(working_directory, "whole_brain_dilated_mask.tif"), dilated_mask.astype("uint8"))

# Process the first half of the image
midline = dilated_mask.shape[-1] // 2
dilated_mask_hemisphere = dilated_mask.copy()
dilated_mask_hemisphere[:, :, midline:] = 0
tifffile.imwrite(os.path.join(working_directory, "hemisphere_dilated_mask.tif"), dilated_mask_hemisphere.astype("uint8"))
