import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tifffile

matplotlib.use("Agg")

saving_directory = r"E:\tto\mapping_aba_to_gubra"
resources_directory = r"C:\Users\MANDOUDNA\PycharmProjects\cmlite\resources"
working_directory = os.path.join(resources_directory, "abc_atlas")
reference_directory = os.path.join(resources_directory, "atlas")
reference = tifffile.imread(os.path.join(reference_directory, "gubra_reference_mouse.tif"))

data_path = os.path.join(saving_directory, r"all_transformed_cells_gubra_5.npy")
# data_path = os.path.join(working_directory, r"all_transformed_cells_gubra_5.npy")
data = np.load(data_path)
print(np.max(data, axis=0))

########################################################################################################################
# HORIZONTAL
########################################################################################################################

reference_coronal = np.swapaxes(reference, 0, 2)

data_horizontal = data.copy()
data_horizontal[:, (1, 2)] = data_horizontal[:, (2, 1)]
data_horizontal[:, (0, 1)] = data_horizontal[:, (1, 0)]

fig = plt.figure()
ax = plt.subplot(111)
ax.scatter(data_horizontal[:, 0], data_horizontal[:, 1], s=0.0001)
ax.imshow(np.max(reference_coronal, 0), cmap="Grays")
plt.savefig(os.path.join(saving_directory, f"all_transformed_cells_gubra_5_horizontal_new.png"), dpi=300)

########################################################################################################################
# SAGITTAL
########################################################################################################################

reference_sagittal = np.swapaxes(reference, 1, 2)
data_sagittal = data.copy()

fig = plt.figure()
ax = plt.subplot(111)
ax.scatter(data_sagittal[:, 0], data_sagittal[:, 1], s=0.0001)
ax.imshow(np.max(reference_sagittal, 0), cmap="Grays")
plt.savefig(os.path.join(saving_directory, f"all_transformed_cells_gubra_5_sagittal_new.png"), dpi=300)
