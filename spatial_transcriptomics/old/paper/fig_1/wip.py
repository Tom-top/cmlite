import os

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import requests
import json

import utils.utils as ut

import analysis.measurements.voxelization as vox

# matplotlib.use("Agg")
matplotlib.use("Qt5Agg")

atlas_used = "aba"
SAVING_DIRECTORY = ut.create_dir(fr"/default/path")  # PERSONAL
DOWNLOAD_BASE = r'E:\tto\spatial_transcriptomics'
TRANSFORM_DIR = r"resources/abc_atlas"
ATLAS_DIR = r"resources\atlas"
ATLAS_FILE = os.path.join(ATLAS_DIR, f"/default/path")  # PERSONAL
REFERENCE_FILE = os.path.join(ATLAS_DIR, f"/default/path")  # PERSONAL
ATLAS = np.swapaxes(tifffile.imread(ATLAS_FILE), 0, 2)
REFERENCE = np.swapaxes(tifffile.imread(REFERENCE_FILE), 0, 2)
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]

datasets = np.arange(1, 5, 1)
transformed_coordinates_all = []
transformed_coordinates_neurons = []

for dataset_n in datasets:
    dataset_id = f"Zhuang-ABCA-{dataset_n}"
    url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
    manifest = json.loads(requests.get(url).text)
    metadata = manifest['file_listing'][dataset_id]['metadata']
    metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
    metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']

    # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
    cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
    cell_metadata_file_ccf = os.path.join(DOWNLOAD_BASE, cell_metadata_path_ccf)
    cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
    cell_metadata_ccf.set_index('cell_label', inplace=True)
    cell_labels = cell_metadata_ccf.index

    # Views
    cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
    cell_metadata_file_views = os.path.join(DOWNLOAD_BASE, cell_metadata_path_views)
    cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
    cell_metadata_views.set_index('cell_label', inplace=True)

    filtered_metadata_views = cell_metadata_views.loc[cell_labels]
    cells_class = filtered_metadata_views["class"].tolist()
    non_neuronal_mask = np.array(
        [True if any([j in i for j in NON_NEURONAL_CELL_TYPES]) else False for i in cells_class])

    if atlas_used == "gubra":
        transformed_coordinates = np.load(os.path.join(TRANSFORM_DIR, f"all_transformed_cells_{dataset_n}.npy"))
    elif atlas_used == "aba":
        scaling = 25 * 1.6
        transformed_coordinates = []
        for n, cl in enumerate(cell_labels):
            cell_data = cell_metadata_ccf.loc[cl]
            x, y, z = int(cell_data["x"] * scaling), int(cell_data["y"] * scaling), int(cell_data["z"] * scaling)
            transformed_coordinates.append([x, y, z])
            print(f"Fetching cell: {n + 1}/{len(filtered_metadata_views)}: x:{x}, y:{y}, z:{z}")
        transformed_coordinates = np.array(transformed_coordinates)
    else:
        ut.CmliteError(f"Please select a valid atlas: aba or gubra not {atlas_used}")

    transformed_coordinates_all.append(transformed_coordinates)
    transformed_coordinates_neurons.append(transformed_coordinates[~non_neuronal_mask])

transformed_coordinates_all_stack = np.vstack(transformed_coordinates_all)
transformed_coordinates_neurons_stack = np.vstack(transformed_coordinates_neurons)

voxelization_parameter = dict(
    shape=np.transpose(tifffile.imread(REFERENCE_FILE), (1, 2, 0)).shape, #(512, 268, 369)
    dtype="uint16",
    weights=None,
    method='sphere',
    radius=np.array([1, 1, 1]),
    kernel=None,
    processes=None,
    verbose=True
)

hm_path_all = os.path.join(SAVING_DIRECTORY, f"heatmap_all_sphere_{atlas_used}.tif")
if os.path.exists(hm_path_all):
    os.remove(hm_path_all)
hm = vox.voxelize(transformed_coordinates_all_stack,
                  **voxelization_parameter)
tifffile.imwrite(hm_path_all, hm)

hm_path_neurons = os.path.join(SAVING_DIRECTORY, f"heatmap_neurons_sphere_{atlas_used}.tif")
if os.path.exists(hm_path_neurons):
    os.remove(hm_path_neurons)
hm = vox.voxelize(transformed_coordinates_neurons_stack,
                  **voxelization_parameter)
tifffile.imwrite(hm_path_neurons, hm)

########################################################################################################################
#
########################################################################################################################

atlas_shape = ATLAS.shape
clipped_atlas = ATLAS.copy()
clipped_atlas[:, :, int(atlas_shape[-1]):] = 0
tifffile.imwrite(os.path.join(SAVING_DIRECTORY, f"clipped_atlas_{atlas_used}"), clipped_atlas)
clipped_reference = REFERENCE.copy()
clipped_reference[:, :, int(atlas_shape[-1]):] = 0
tifffile.imwrite(os.path.join(SAVING_DIRECTORY, f"clipped_reference_{atlas_used}"), clipped_reference)
unique_atlas_values = np.unique(clipped_atlas)
clipped_atlas_mask = clipped_atlas > 0

hm_all = np.swapaxes(tifffile.imread(hm_path_all), 0, 1)
# tifffile.imwrite(os.path.join(SAVING_DIRECTORY, "hm.tif"), hm_all)
min_all, max_all = np.min(hm_all[clipped_atlas_mask]), np.max(hm_all[clipped_atlas_mask])
avg_all, std_all = np.mean(hm_all[clipped_atlas_mask]), np.std(hm_all[clipped_atlas_mask])
total_all_counts = []
avg_all_counts = []

# [OPTIONAL] Create the mask of a specific region
# reg_n, reg_id = "MB", 5808
# test = hm_all.copy()
# reg_mask = clipped_atlas == reg_id
# test[~reg_mask] = 0
# tifffile.imwrite(os.path.join(SAVING_DIRECTORY, f"{reg_n}_test.tif"), test)

for reg in unique_atlas_values:
    reg_mask = clipped_atlas == reg
    total_all_count = np.sum(hm_all[reg_mask])
    total_all_counts.append(total_all_count)
    print(f"{total_all_count} neurons found in {reg}")
    avg_all_count = np.mean(hm_all[reg_mask])
    print(f"Average neurons in {reg}: {avg_all_count}")
    avg_all_counts.append(avg_all_count)

hm_neurons = np.swapaxes(tifffile.imread(hm_path_neurons), 0, 1)
min_neurons, max_neurons = np.min(hm_neurons[clipped_atlas_mask]), np.max(hm_neurons[clipped_atlas_mask])
avg_neurons, std_neurons = np.mean(hm_neurons[clipped_atlas_mask]), np.std(hm_neurons[clipped_atlas_mask])
total_neuron_counts = []
avg_neuron_counts = []
region_voxel_size = []

for reg in unique_atlas_values:
    reg_mask = clipped_atlas == reg
    total_neuron_count = np.sum(hm_neurons[reg_mask])
    total_neuron_counts.append(total_neuron_count)
    print(f"{total_neuron_count} neurons found in {reg}")
    avg_neuron_count = np.mean(hm_neurons[reg_mask])
    print(f"Average neurons in {reg}: {avg_neuron_count}")
    region_voxel_size.append(len(hm_neurons[reg_mask]))
    avg_neuron_counts.append(avg_neuron_count)

########################################################################################################################
# LOOP OVER THE SELECTED ATLAS PLANES
########################################################################################################################

selected_atlas_idx = [85, 145, 205, 265, 325, 385, 445]
clipped_atlas_coronal = np.swapaxes(clipped_atlas, 0, 1)

# Get the viridis colormap
cmap = plt.cm.viridis

# Normalize the neuron counts to the range [0, 1] for colormap mapping
hm_neurons = tifffile.imread(hm_path_neurons)
min_neuron_count = np.min(hm_neurons)
max_neuron_count = 4
norm = plt.Normalize(vmin=min_neuron_count, vmax=max_neuron_count)

for atlas_idx in selected_atlas_idx:
    atlas_slice = clipped_atlas_coronal[atlas_idx]
    atlas_slice_rgb = np.zeros((*atlas_slice.shape, 3), dtype=np.uint8)  # RGB image
    unique_ids = np.unique(atlas_slice)
    for ui in unique_ids:
        if ui != 0:
            reg_mask = atlas_slice == ui
            # Calculate the average neuron count in this region
            average_neuron_count = np.mean(hm_neurons[atlas_idx][reg_mask])

            # Map the average neuron count to the colormap
            color = cmap(norm(average_neuron_count))[:3]  # Get RGB values, ignore alpha

            # Assign the RGB color to the corresponding area in the RGB slice
            atlas_slice_rgb[reg_mask] = (np.array(color) * 255).astype(np.uint8)
    tifffile.imwrite(os.path.join(SAVING_DIRECTORY, f"neuronal_heatmap_{atlas_idx}_{atlas_used}.tif"), atlas_slice_rgb)
    tifffile.imwrite(os.path.join(SAVING_DIRECTORY, f"atlas_16b_{atlas_idx}_{atlas_used}.tif"), atlas_slice)


########################################################################################################################
# Plot the cell density across brain regions
########################################################################################################################


atlas_metadata_path = os.path.join(ATLAS_DIR, f"{atlas_used}_annotation_mouse.json")
atlas_metadata = load_json_file(atlas_metadata_path)

bar_plot_data = [[total_neuron_counts, total_all_counts], [avg_neuron_counts, avg_all_counts]]

clipped_atlas_coronal = np.swapaxes(clipped_atlas, 0, 1)
hm_all = tifffile.imread(hm_path_all)
avg_cell_density = np.mean(hm_all[clipped_atlas_coronal > 0])
hm_neurons = tifffile.imread(hm_path_neurons)
avg_neuronal_density = np.mean(hm_neurons[clipped_atlas_coronal > 0])

for dn, d in enumerate(bar_plot_data):

    if dn == 0:
        saving_name = "total_number"
    else:
        saving_name = "density"

    # Neurons
    sorted_avg_neuron_counts, sorted_avg_neuron_idx = sort_cells_and_get_indices(d[0])
    sorted_unique_atlas_values = unique_atlas_values[sorted_avg_neuron_idx]

    sorted_unique_atlas_acro = [find_dict_by_id_value(atlas_metadata, i).get("acronym")
                                if find_dict_by_id_value(atlas_metadata, i)
                                else None for i in sorted_unique_atlas_values]
    sorted_unique_atlas_hex = [find_dict_by_id_value(atlas_metadata, i).get("color_hex_triplet")
                               if find_dict_by_id_value(atlas_metadata, i)
                               else None for i in sorted_unique_atlas_values]
    none_mask = np.array(sorted_unique_atlas_acro) == None
    sorted_unique_atlas_acro = np.array(sorted_unique_atlas_acro)[~none_mask]
    sorted_unique_atlas_hex = np.array(sorted_unique_atlas_hex)[~none_mask]
    sorted_unique_atlas_hex = np.array(['#' + color for color in sorted_unique_atlas_hex])

    # Create a bar plot
    n_bars = -1
    x_tick_fs = 6
    plt.figure(figsize=(15, 3))  # Adjust the figure size as needed
    bars = plt.bar(sorted_unique_atlas_acro[:n_bars],
                   np.array(sorted_avg_neuron_counts)[~none_mask][:n_bars],
                   color=sorted_unique_atlas_hex[:n_bars], linewidth=0.1, edgecolor='black',
                   width=1)
    # Add a horizontal line at the average value
    plt.axhline(y=avg_neuronal_density, color='red', linestyle='--', linewidth=1.5,
                label=f'Average: {avg_neuronal_density:.2f}')
    # plt.title('Cell Counts in Brain Regions')
    # plt.xlabel('Brain Region')
    plt.ylabel(f'{saving_name} neurons')
    plt.xticks(rotation=45, ha="right", fontsize=x_tick_fs)
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(os.path.join(SAVING_DIRECTORY, f"{saving_name}_neurons_top_{n_bars}_{atlas_used}.png"), dpi=300)
    plt.savefig(os.path.join(SAVING_DIRECTORY, f"{saving_name}_neurons_top_{n_bars}_{atlas_used}.svg"), dpi=300)
    # plt.show()

    # All cells
    sorted_avg_all_counts, sorted_avg_all_idx = sort_cells_and_get_indices(d[1])
    sorted_unique_atlas_values = unique_atlas_values[sorted_avg_all_idx]

    sorted_unique_atlas_acro = [find_dict_by_id_value(atlas_metadata, i).get("acronym")
                                if find_dict_by_id_value(atlas_metadata, i)
                                else None for i in sorted_unique_atlas_values]
    sorted_unique_atlas_hex = [find_dict_by_id_value(atlas_metadata, i).get("color_hex_triplet")
                               if find_dict_by_id_value(atlas_metadata, i)
                               else None for i in sorted_unique_atlas_values]
    none_mask = np.array(sorted_unique_atlas_acro) == None
    sorted_unique_atlas_acro = np.array(sorted_unique_atlas_acro)[~none_mask]
    sorted_unique_atlas_hex = np.array(sorted_unique_atlas_hex)[~none_mask]
    sorted_unique_atlas_hex = np.array(['#' + color for color in sorted_unique_atlas_hex])

    # Create a bar plot
    plt.figure(figsize=(15, 3))  # Adjust the figure size as needed
    bars = plt.bar(sorted_unique_atlas_acro[:n_bars],
                   np.array(sorted_avg_all_counts)[~none_mask][:n_bars],
                   color=sorted_unique_atlas_hex[:n_bars], linewidth=0.1, edgecolor='black',
                   width=1)
    # Add a horizontal line at the average value
    plt.axhline(y=avg_cell_density, color='red', linestyle='--', linewidth=1.5,
                label=f'Average: {avg_cell_density:.2f}')
    # plt.title('Cell Counts in Brain Regions')
    # plt.xlabel('Brain Region')
    plt.ylabel(f'{saving_name} cells')
    plt.xticks(rotation=45, ha="right", fontsize=x_tick_fs)
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(os.path.join(SAVING_DIRECTORY, f"{saving_name}_all_top_{n_bars}_{atlas_used}.png"), dpi=300)
    plt.savefig(os.path.join(SAVING_DIRECTORY, f"{saving_name}_all_top_{n_bars}_{atlas_used}.svg"), dpi=300)
    # plt.show()


########################################################################################################################
# PLOT METAREGION COUNTS
########################################################################################################################

def find_children_ids(data, id):
    # Main function to gather all ids recursively
    def gather_ids(data):
        ids = [data['id']]
        for child in data.get('children', []):
            ids.extend(gather_ids(child))
        return ids

    # Find the node with the given id
    id_data = find_dict_by_id_value(data, id, key="id")

    # Gather all ids starting from the found node
    if id_data and id_data['id'] == id:
        return gather_ids(id_data)[1:]  # [1:] to exclude the id of the current node
    else:
        return None


# bar_plot_data = [[total_neuron_counts, total_all_counts], [avg_neuron_counts, avg_all_counts]]
if atlas_used == "gubra":
    metaregion_ids = [5005, 5379, 5454, 5555, 5571, 5610, 5643, 5717, 5808, 5885, 5937, 6016, 6103, 6294]
elif atlas_used == "aba":
    metaregion_ids = [315, 698, 1089, 703, 477, 803, 549, 1097, 313, 771, 354, 512, 1009, 73]
metaregion_neuronal_densities = []

for metaregion in metaregion_ids:
    metaregion_acronym = find_dict_by_id_value(atlas_metadata, metaregion, key="id")["acronym"]
    metaregion_children = find_children_ids(atlas_metadata, metaregion)
    metaregion_children_mask = np.array([True if i in metaregion_children else False for i in unique_atlas_values])
    metaregion_counts = np.array(total_neuron_counts)[metaregion_children_mask]
    metaregion_sizes = np.array(region_voxel_size)[metaregion_children_mask]
    metaregion_neuronal_density = np.sum(metaregion_counts) / np.sum(metaregion_sizes)
    metaregion_neuronal_densities.append(metaregion_neuronal_density)
    ut.print_c(f"[INFO] Neuronal density in {metaregion_acronym}: {metaregion_neuronal_density}")

# Neurons
metaregion_acro = [find_dict_by_id_value(atlas_metadata, i).get("acronym")
                   if find_dict_by_id_value(atlas_metadata, i)
                   else None for i in metaregion_ids]
metaregion_hex = [find_dict_by_id_value(atlas_metadata, i).get("color_hex_triplet")
                  if find_dict_by_id_value(atlas_metadata, i)
                  else None for i in metaregion_ids]
metaregion_hex = np.array(['#' + color for color in metaregion_hex])

# Create a bar plot
plt.figure(figsize=(7, 3))  # Adjust the figure size as needed
bars = plt.bar(metaregion_acro,
               np.array(metaregion_neuronal_densities),
               color=metaregion_hex, linewidth=0.5, edgecolor='black',
               width=1)
plt.ylabel(f'density cells')
plt.xticks(rotation=45, ha="right", fontsize=x_tick_fs)
plt.tight_layout()  # Adjust layout to not cut off labels
plt.savefig(os.path.join(SAVING_DIRECTORY, f"density_metaregions_{atlas_used}.png"), dpi=300)
plt.savefig(os.path.join(SAVING_DIRECTORY, f"density_metaregions_{atlas_used}.svg"), dpi=300)
# plt.show()

########################################################################################################################
#
########################################################################################################################

sorted_unique_atlas_ids = [find_dict_by_id_value(atlas_metadata, i).get("id")
                           if find_dict_by_id_value(atlas_metadata, i)
                           else None for i in sorted_unique_atlas_values]
sorted_unique_atlas_ids = np.array(sorted_unique_atlas_ids)[~none_mask]

metastructure_names = ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", "TH", "HY", "MB", "P", "MY", "CB",
                       "fiber tracts", "VS"]
final_id = 6306

for dn, d in enumerate([hm_all, hm_neurons]):

    if dn == 0:
        saving_name = "all_cells"
    else:
        saving_name = "neurons"

    voxel_values_for_all_regions = []
    colors_for_all_regions = []

    for m in range(len(metastructure_names)):
        voxel_values_for_metaregion = []
        if m < len(metastructure_names) - 1:
            id_s = find_dict_by_id_value(atlas_metadata, metastructure_names[m], key="acronym")["id"]
            id_e = find_dict_by_id_value(atlas_metadata, metastructure_names[m + 1], key="acronym")["id"]
        else:
            id_s, id_e = 6294, final_id
        # metastructure_id_range.append([id_s, id_e-1])
        print(id_s, id_e)
        for reg in np.arange(id_s, id_e, 1):
            reg_mask = clipped_atlas == reg
            voxel_values_in_region = d[reg_mask]
            voxel_values_for_metaregion.append(voxel_values_in_region)
        voxel_values_for_all_regions.append(np.concatenate(voxel_values_for_metaregion))
        colors_for_all_regions.append(find_dict_by_id_value(atlas_metadata,
                                                            metastructure_names[m], key="acronym")["color_hex_triplet"])
    colors_for_all_regions = np.array(['#' + color for color in colors_for_all_regions])

    hist_range = np.array([0, 20])
    fig = plt.figure()
    ax = plt.subplot(111)
    # Generate the stacked histogram
    ax.hist(voxel_values_for_all_regions, bins=hist_range[1] - hist_range[0], stacked=True,
            color=colors_for_all_regions, label=metastructure_names,
            range=hist_range, log=False, linewidth=0.2, edgecolor='black')
    ax.legend(ncol=2)
    # Add titles and labels
    # ax.title('Distribution of Cell Counts by Brain Region')
    ax.set_xlabel('Cell Counts')
    ax.set_xticks(np.arange(hist_range[0] + 0.5, hist_range[1] + 0.5, 2))
    ax.set_xticklabels(np.arange(hist_range[0], hist_range[1], 2))
    ax.set_xlim(hist_range)
    ax.set_ylabel('Frequency')
    ax.set_ylim(0, 4.7 * 10 ** (6))
    plt.savefig(os.path.join(SAVING_DIRECTORY, f"{saving_name}_density_distribution_{atlas_used}.png"), dpi=300)
    plt.savefig(os.path.join(SAVING_DIRECTORY, f"{saving_name}_density_distribution_{atlas_used}.svg"), dpi=300)
    # plt.show()

# fig = plt.figure()
# ax = plt.subplot(111)
# # Plotting the cumulative density for each region
# for region, counts, color in zip(metastructure_names, voxel_values_for_all_regions, colors_for_all_regions):
#     sorted_counts = np.sort(counts)
#     yvals = np.arange(len(sorted_counts)) / float(len(sorted_counts) - 1)
#     ax.plot(sorted_counts, yvals, label=region, color=color)
#
# ax.set_xlim(0, 20)
# plt.xlabel('Cell Counts')
# plt.ylabel('Cumulative Density')
# plt.title('Cumulative Density of Cell Counts Across Brain Regions')
# plt.legend()
# plt.grid(True)
#
# plt.show()
