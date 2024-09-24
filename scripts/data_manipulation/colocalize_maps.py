import os

import numpy as np
import tifffile


def calculate_overlap(array1, array2):
    """
    Calculates the overlap between two 3D arrays.
    Both arrays must have the same shape.

    Parameters:
    array1 (numpy.ndarray): First 3D array (values between 0-1).
    array2 (numpy.ndarray): Second 3D array (values between 0-1).

    Returns:
    numpy.ndarray: A new 3D array representing the overlap.
    """
    # Ensure both arrays have the same shape
    if array1.shape != array2.shape:
        raise ValueError("Both arrays must have the same shape.")

    # Calculate the overlap by element-wise multiplication
    overlap = np.multiply(array1, array2)

    return overlap


saving_dir = r"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results\gene_expression"

# Example usage:
# Create two random 3D arrays with values between 0 and 1
array1 = tifffile.imread(r"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results\gene_expression\heatmap_Glp1r_neurons_dynamic_bin.tif")
norm_array1 = (array1 - array1.min()) / (array1.max() - array1.min())
array2 = tifffile.imread(r"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results\gene_expression\heatmap_Gipr_neurons_dynamic_bin.tif")
norm_array2 = (array2 - array2.min()) / (array2.max() - array2.min())

# # Set random values to 1 or 0
# norm_array1 = np.round(array1)
# norm_array2 = np.round(array2)

# Calculate the overlap
result = calculate_overlap(norm_array1, norm_array2)

tifffile.imwrite(os.path.join(saving_dir, "glp1r_merfish_gipr_merfish_overlap.tif"), result)
