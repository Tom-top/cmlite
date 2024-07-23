import os

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import requests
import json

from ClearMap.Environment import *

# matplotlib.use("Agg")
matplotlib.use("Qt5Agg")

saving_directory = "/mnt/data/spatial_transcriptomics/results/whole_brain/whole_brain/transcriptomics/heatmap"
atlas_file = r"/mnt/data/spatial_transcriptomics/atlas_ressources/gubra_annotations_coronal.tif"
reference_file = r"/mnt/data/spatial_transcriptomics/atlas_ressources/gubra_template_coronal.tif"
atlas = tifffile.imread(atlas_file)
reference = tifffile.imread(reference_file)

datasets = np.arange(1, 5, 1)
transformed_coordinates_all = []
transformed_coordinates_neurons = []

for dataset_n in datasets:
    dataset_id = f"Zhuang-ABCA-{dataset_n}"
    download_base = r'/mnt/data/spatial_transcriptomics'
    url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
    manifest = json.loads(requests.get(url).text)
    metadata = manifest['file_listing'][dataset_id]['metadata']
    metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
    metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']

    # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
    cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
    cell_metadata_file_ccf = os.path.join(download_base, cell_metadata_path_ccf)
    cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
    cell_metadata_ccf.set_index('cell_label', inplace=True)
    cell_labels = cell_metadata_ccf.index

    # Views
    cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
    cell_metadata_file_views = os.path.join(download_base, cell_metadata_path_views)
    cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
    cell_metadata_views.set_index('cell_label', inplace=True)

    filtered_metadata_views = cell_metadata_views.loc[cell_labels]
    cells_class = filtered_metadata_views["class"].tolist()
    non_neuronal_cell_types = ["Astro", "Oligo", "Vascular", "Immune", "Epen"]
    neuronal_mask = np.array([True if any([j in i for j in non_neuronal_cell_types]) else False for i in cells_class])

    # Filter out the cells
    transformed_coordinates = np.load(r"/mnt/data/spatial_transcriptomics/results/transformed_cells_to_gubra/"
                                      fr"general/all_transformed_cells_{dataset_n}.npy")

    transformed_coordinates_all.append(transformed_coordinates)
    transformed_coordinates_neurons.append(transformed_coordinates[~neuronal_mask])

transformed_coordinates_all_stack = np.vstack(transformed_coordinates_all)
transformed_coordinates_neurons_stack = np.vstack(transformed_coordinates_neurons)

voxelization_parameter = dict(
            shape=(512, 268, 369),
            dtype="uint16",
            weights=None,
            method='sphere',
            # method='rectangle',
            # radius=np.array([5, 5, 5]),
            radius=np.array([1, 1, 1]),
            kernel=None,
            processes=None,
            verbose=True
        )

hm_path_all = os.path.join(saving_directory, f"heatmap_all_sphere.tif")
if os.path.exists(hm_path_all):
    os.remove(hm_path_all)
hm = vox.voxelize(transformed_coordinates_all_stack,
                  **voxelization_parameter)
tifffile.imwrite(hm_path_all, hm)

hm_path_neurons = os.path.join(saving_directory, f"heatmap_neurons_sphere.tif")
if os.path.exists(hm_path_neurons):
    os.remove(hm_path_neurons)
hm = vox.voxelize(transformed_coordinates_neurons_stack,
                  **voxelization_parameter)
tifffile.imwrite(hm_path_neurons, hm)

########################################################################################################################
#
########################################################################################################################

clipped_atlas = atlas.copy()
clipped_atlas[:, :, 190:] = 0
tifffile.imwrite(os.path.join(saving_directory, "clipped_atlas.tif"), clipped_atlas)
clipped_reference = reference.copy()
clipped_reference[:, :, 190:] = 0
tifffile.imwrite(os.path.join(saving_directory, "clipped_reference.tif"), clipped_reference)
unique_atlas_values = np.unique(clipped_atlas)
clipped_atlas_mask = clipped_atlas > 0

hm_all = tifffile.imread(hm_path_all)
min_all, max_all = np.min(hm_all[clipped_atlas_mask]), np.max(hm_all[clipped_atlas_mask])
avg_all, std_all = np.mean(hm_all[clipped_atlas_mask]), np.std(hm_all[clipped_atlas_mask])
total_all_counts = []
avg_all_counts = []

# [OPTIONAL] Create the mask of a specific region
reg_n, reg_id = "MB", 5808
test = hm_all.copy()
reg_mask = clipped_atlas == reg_id
test[~reg_mask] = 0
print(test[reg_mask])
print(np.mean(test[reg_mask]))
tifffile.imwrite(os.path.join(saving_directory, f"{reg_n}_test.tif"), test)

for reg in unique_atlas_values:
    reg_mask = clipped_atlas == reg
    total_all_count = np.sum(hm_all[reg_mask])
    total_all_counts.append(total_all_count)
    print(f"{total_all_count} neurons found in {reg}")
    avg_all_count = np.mean(hm_all[reg_mask])
    print(f"Average neurons in {reg}: {avg_all_count}")
    avg_all_counts.append(avg_all_count)

hm_neurons = tifffile.imread(hm_path_neurons)
min_neurons, max_neurons = np.min(hm_neurons[clipped_atlas_mask]), np.max(hm_neurons[clipped_atlas_mask])
avg_neurons, std_neurons = np.mean(hm_neurons[clipped_atlas_mask]), np.std(hm_neurons[clipped_atlas_mask])
total_neuron_counts = []
avg_neuron_counts = []

for reg in unique_atlas_values:
    reg_mask = clipped_atlas == reg
    total_neuron_count = np.sum(hm_neurons[reg_mask])
    total_neuron_counts.append(total_neuron_count)
    print(f"{total_neuron_count} neurons found in {reg}")
    avg_neuron_count = np.mean(hm_neurons[reg_mask])
    print(f"Average neurons in {reg}: {avg_neuron_count}")
    avg_neuron_counts.append(avg_neuron_count)

########################################################################################################################
# Plot the cell density across brain regions
########################################################################################################################

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Attempt to load the JSON content
            return json.load(file)
    except FileNotFoundError:
        print("Error: The file was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def find_dict_by_id_value(data, target_id, key="id"):
    # If data is a dictionary, check if it contains the key with the matching target value
    if isinstance(data, dict):
        if key in data and data[key] == target_id:
            # Return the entire dictionary if key with matching value is found
            return data
        else:
            # Otherwise, search recursively in values
            for v in data.values():
                result = find_dict_by_id_value(v, target_id, key=key)
                if result is not None:
                    return result

    # If data is a list, iterate and search recursively in each item
    elif isinstance(data, list):
        for item in data:
            result = find_dict_by_id_value(item, target_id, key=key)
            if result is not None:
                return result

    # Return None if the key with the matching value is not found in any dictionary
    return None


def sort_cells_and_get_indices(cell_counts):
    # Convert the list to a numpy array for easier manipulation
    cell_counts_array = np.array(cell_counts)
    # Get indices that would sort the array from lowest to highest
    ascending_indices = np.argsort(cell_counts_array)
    # Reverse the indices to get highest to lowest
    descending_indices = ascending_indices[::-1]
    # Sort the cell counts array using the descending indices
    sorted_cell_counts = cell_counts_array[descending_indices]

    return sorted_cell_counts, descending_indices

atlas_ressources_path = "/home/imaging/PycharmProjects/spatial_transcriptomics/ClearMap/Resources/Atlas"
atlas_metadata_gubra_path = os.path.join(atlas_ressources_path, "Gubra_annotation.json")

atlas_metadata_gubra = load_json_file(atlas_metadata_gubra_path)

bar_plot_data = [[total_neuron_counts, total_all_counts], [avg_neuron_counts, avg_all_counts]]

for dn, d in enumerate(bar_plot_data):

    if dn == 0:
        saving_name = "total_number"
    else:
        saving_name = "density"

    # Neurons
    sorted_avg_neuron_counts, sorted_avg_neuron_idx = sort_cells_and_get_indices(d[0])
    sorted_unique_atlas_values = unique_atlas_values[sorted_avg_neuron_idx]

    sorted_unique_atlas_acro = [find_dict_by_id_value(atlas_metadata_gubra, i).get("acronym")
                                if find_dict_by_id_value(atlas_metadata_gubra, i)
                                else None for i in sorted_unique_atlas_values]
    sorted_unique_atlas_hex = [find_dict_by_id_value(atlas_metadata_gubra, i).get("color_hex_triplet")
                                if find_dict_by_id_value(atlas_metadata_gubra, i)
                                else None for i in sorted_unique_atlas_values]
    none_mask = np.array(sorted_unique_atlas_acro) == None
    sorted_unique_atlas_acro = np.array(sorted_unique_atlas_acro)[~none_mask]
    sorted_unique_atlas_hex = np.array(sorted_unique_atlas_hex)[~none_mask]
    sorted_unique_atlas_hex = np.array(['#' + color for color in sorted_unique_atlas_hex])

    # Create a bar plot
    n_bars = 50
    x_tick_fs = 6
    plt.figure()  # Adjust the figure size as needed
    bars = plt.bar(sorted_unique_atlas_acro[:n_bars],
                   np.array(sorted_avg_neuron_counts)[~none_mask][:n_bars],
                   color=sorted_unique_atlas_hex[:n_bars], linewidth=0.2, edgecolor='black',
                   width=1)
    # plt.title('Cell Counts in Brain Regions')
    # plt.xlabel('Brain Region')
    plt.ylabel(f'{saving_name} neurons')
    plt.xticks(rotation=45, ha="right", fontsize=x_tick_fs)
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(os.path.join(saving_directory, f"{saving_name}_neurons_top_{n_bars}.png"), dpi=300)
    plt.savefig(os.path.join(saving_directory, f"{saving_name}_neurons_top_{n_bars}.svg"), dpi=300)
    # plt.show()

    # All cells
    sorted_avg_all_counts, sorted_avg_all_idx = sort_cells_and_get_indices(d[1])
    sorted_unique_atlas_values = unique_atlas_values[sorted_avg_all_idx]

    sorted_unique_atlas_acro = [find_dict_by_id_value(atlas_metadata_gubra, i).get("acronym")
                                if find_dict_by_id_value(atlas_metadata_gubra, i)
                                else None for i in sorted_unique_atlas_values]
    sorted_unique_atlas_hex = [find_dict_by_id_value(atlas_metadata_gubra, i).get("color_hex_triplet")
                                if find_dict_by_id_value(atlas_metadata_gubra, i)
                                else None for i in sorted_unique_atlas_values]
    none_mask = np.array(sorted_unique_atlas_acro) == None
    sorted_unique_atlas_acro = np.array(sorted_unique_atlas_acro)[~none_mask]
    sorted_unique_atlas_hex = np.array(sorted_unique_atlas_hex)[~none_mask]
    sorted_unique_atlas_hex = np.array(['#' + color for color in sorted_unique_atlas_hex])

    # Create a bar plot
    plt.figure()  # Adjust the figure size as needed
    bars = plt.bar(sorted_unique_atlas_acro[:n_bars],
                   np.array(sorted_avg_all_counts)[~none_mask][:n_bars],
                   color=sorted_unique_atlas_hex[:n_bars], linewidth=0.2, edgecolor='black',
                   width=1)
    # plt.title('Cell Counts in Brain Regions')
    # plt.xlabel('Brain Region')
    plt.ylabel(f'{saving_name} cells')
    plt.xticks(rotation=45, ha="right", fontsize=x_tick_fs)
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(os.path.join(saving_directory, f"{saving_name}_all_top_{n_bars}.png"), dpi=300)
    plt.savefig(os.path.join(saving_directory, f"{saving_name}_all_top_{n_bars}.svg"), dpi=300)
    # plt.show()

########################################################################################################################
#
########################################################################################################################

sorted_unique_atlas_ids = [find_dict_by_id_value(atlas_metadata_gubra, i).get("id")
                                if find_dict_by_id_value(atlas_metadata_gubra, i)
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
            id_s = find_dict_by_id_value(atlas_metadata_gubra, metastructure_names[m], key="acronym")["id"]
            id_e = find_dict_by_id_value(atlas_metadata_gubra, metastructure_names[m+1], key="acronym")["id"]
        else:
            id_s, id_e = 6294, final_id
        # metastructure_id_range.append([id_s, id_e-1])
        print(id_s, id_e)
        for reg in np.arange(id_s, id_e, 1):
            reg_mask = clipped_atlas == reg
            voxel_values_in_region = d[reg_mask]
            voxel_values_for_metaregion.append(voxel_values_in_region)
        voxel_values_for_all_regions.append(np.concatenate(voxel_values_for_metaregion))
        colors_for_all_regions.append(find_dict_by_id_value(atlas_metadata_gubra,
                                                            metastructure_names[m], key="acronym")["color_hex_triplet"])
    colors_for_all_regions = np.array(['#' + color for color in colors_for_all_regions])

    hist_range = np.array([0, 20])
    fig = plt.figure()
    ax = plt.subplot(111)
    # Generate the stacked histogram
    ax.hist(voxel_values_for_all_regions, bins=hist_range[1]-hist_range[0], stacked=True,
            color=colors_for_all_regions, label=metastructure_names,
            range=hist_range, log=False, linewidth=0.2, edgecolor='black')
    ax.legend(ncol=2)
    # Add titles and labels
    # ax.title('Distribution of Cell Counts by Brain Region')
    ax.set_xlabel('Cell Counts')
    ax.set_xticks(np.arange(hist_range[0]+0.5, hist_range[1]+0.5, 2))
    ax.set_xticklabels(np.arange(hist_range[0], hist_range[1], 2))
    ax.set_xlim(hist_range)
    ax.set_ylabel('Frequency')
    ax.set_ylim(0, 4.7*10**(6))
    plt.savefig(os.path.join(saving_directory, f"{saving_name}_density_distribution.png"), dpi=300)
    plt.savefig(os.path.join(saving_directory, f"{saving_name}_density_distribution.svg"), dpi=300)
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