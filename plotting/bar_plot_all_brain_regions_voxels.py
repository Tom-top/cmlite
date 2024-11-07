import os
import json

import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut


def find_region_by_id(region_data, target_id):
    if isinstance(region_data, dict):
        if region_data.get('id') == target_id:
            return region_data
        for child in region_data.get('children', []):
            result = find_region_by_id(child, target_id)
            if result:
                return result
    elif isinstance(region_data, list):
        for item in region_data:
            result = find_region_by_id(item, target_id)
            if result:
                return result
    return None


atlas_path = r"resources\atlas"

with open(os.path.join(atlas_path, "gubra_annotation_mouse.json"), "r") as f:
    metadata = json.load(f)
    metadata = metadata["msg"][0]
#
# target_genes = [
#                 # "Mc4r", "Pomc",
#                 # "Mchr1",
#                 # "Npy", "Npy1r", "Npy2r", "Npy4r", "Npy5r", "Npy6r",
#                 "Trem2", "Glp1r",
#                 "Bdnf", "Bdnf_Glp1r",
#                 "Ntrk2", "Ntrk2_Glp1r",
# ]
target_genes = ["Fos", "Npas4", "Nr4a1", "Arc", "Egr1", "Bdnf", "Pcsk1", "Crem", "Igf1", "Scg2", "Nptx2", "Homer1",
                "Pianp", "Serpinb2", "Ostn"]

data_scaling = ["linear", "log2"]
data_scaling = ["linear"]
sort = False

for map_name in target_genes:

    map_color = ""
    working_directory = fr"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results\gene_expression\{map_name}"

    df = pd.read_csv(os.path.join(working_directory, "mean_expression_data.csv"))

    first_column = df.columns[1]  # Get the first column name after 'cluster'
    # Check if a "co-expression" column exists
    if 'co-expression' in df.columns:
        # Sort by the "co-expression" column if it exists
        df_sorted = df.sort_values(by="co-expression", ascending=False)
    else:
        # Otherwise, sort by the first column (assumed to be "Glp1r" here)
        df_sorted = df.sort_values(by=first_column, ascending=False)

    # Limit to top 50 rows after sorting
    df_sorted = df_sorted.head(50)

    # Get the first column dynamically for the x-axis (e.g., 'cluster')
    first_column_x_axis = df.columns[0]  # Use the first column for the x-axis

    # Plot the data
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 5))

    # Plot all regions with their corresponding values and cluster colors
    bars = ax.bar(df_sorted[first_column_x_axis], 2**df_sorted[first_column],
                  color=df_sorted['cluster_color'], width=0.8)

    # Add a black horizontal line representing the mean of the sorted column
    mean_value = 2**df_sorted[first_column].mean()
    ax.axhline(mean_value, color='black', linestyle='--', linewidth=1)

    # Set labels and layout
    ax.set_xlabel('Clusters')
    ax.set_ylabel(f'{first_column} Expression: CPM')

    # Rotate the x-tick labels for better visibility
    ax.set_xticks(range(len(df_sorted[first_column_x_axis])))
    ax.set_xticklabels(df_sorted[first_column_x_axis], rotation=45, ha='right', fontsize=10)

    # Adjust the layout
    fig.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(working_directory, f"{first_column}_expression_per_cluster_top50_with_labels.png"),
                dpi=300)
    plt.savefig(os.path.join(working_directory, f"{first_column}_expression_per_cluster_top50_with_labels.svg"),
                dpi=300)

    # Plot the data
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 5))

    # Plot all regions with their corresponding values and cluster colors
    bars = ax.bar(df[first_column_x_axis], 2 ** df[first_column],
                  color=df['cluster_color'], width=0.8)

    # Add a black horizontal line representing the mean of the sorted column
    mean_value = 2 ** df[first_column].mean()
    ax.axhline(mean_value, color='black', linestyle='--', linewidth=1)

    # Set labels and layout
    ax.set_xlabel('Clusters')
    ax.set_ylabel(f'{first_column} Expression: CPM')

    # Rotate the x-tick labels for better visibility
    ax.set_xticks(range(len(df[first_column_x_axis])))
    ax.set_xticklabels(df[first_column_x_axis], rotation=45, ha='right', fontsize=10)

    # Adjust the layout
    fig.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(working_directory, f"{first_column}_expression_per_cluster_with_labels.png"),
                dpi=300)
    plt.savefig(os.path.join(working_directory, f"{first_column}_expression_per_cluster_with_labels.svg"),
                dpi=300)

    for scaling in data_scaling:
        scaling_directory = os.path.join(working_directory, scaling)
        data_range = np.load(os.path.join(scaling_directory, f"min_max_{map_name}_dynamic.npy"))
        saving_directory = ut.create_dir(os.path.join(scaling_directory, "voxel_wise_intensity_distribution"))
        for dtype in ["_raw", ""]:
            transformed_heatmap_path = os.path.join(scaling_directory, f"heatmap_{map_name}_neurons_dynamic{dtype}_bin.tif")
            transformed_heatmap = tifffile.imread(transformed_heatmap_path)

            transformed_heatmap_sag = np.swapaxes(transformed_heatmap, 0, 2)
            transformed_heatmap_sag = np.swapaxes(transformed_heatmap_sag, 1, 2)

            annotation = tifffile.imread(os.path.join(atlas_path, "gubra_annotation_mouse.tif"))
            unique_labels = sorted(np.unique(annotation))
            print(f"[INFO] {len(unique_labels)} unique labels detected")

            # Initialize dictionaries to hold counts for left and right sides
            mean_voxel_intensity = {}
            colors = {}
            ordered_acronyms = []
            hemisphere = True

            # Count pixels for each label
            for label in unique_labels:

                if label == 0:
                    continue  # Assuming 0 is the background or unlabeled region
                region_info = find_region_by_id(metadata, label)

                if region_info is None:
                    print(f"[WARNING] skipping region with label: {label}")
                else:
                    region_mask = annotation == label
                    region_volume = np.sum(region_mask)
                    region_name = region_info["name"]
                    region_acronym = region_info["acronym"]
                    region_color = f'#{region_info["color_hex_triplet"]}'

                    ordered_acronyms.append(region_acronym)
                    mask_voxels_in_region = transformed_heatmap_sag[region_mask]
                    mask_voxels_in_region_scaled = mask_voxels_in_region * data_range[-1]
                    if hemisphere:
                        transformed_voxels_in_region_mean = np.mean(mask_voxels_in_region_scaled) * 2
                    else:
                        transformed_voxels_in_region_mean = np.mean(mask_voxels_in_region_scaled)
                    mean_voxel_intensity[region_acronym] = transformed_voxels_in_region_mean
                    colors[region_acronym] = region_color

            # Prepare data for plotting
            regions = [region for region in ordered_acronyms if region in mean_voxel_intensity]
            count_values = [mean_voxel_intensity[region] for region in regions]
            bar_colors = [colors[region] for region in regions]

            regions = np.array(regions)
            count_values = np.array(count_values)
            bar_colors = np.array(bar_colors)

            sorted_indices = np.argsort(count_values)
            sorted_regions = regions[sorted_indices]
            sorted_count_values = count_values[sorted_indices]
            sorted_bar_colors = bar_colors[sorted_indices]

            mean_counts = np.mean(count_values)

            ########################################################################################################################
            # Bar plot: density all brain regions rostro-caudal
            ########################################################################################################################

            if map_color:
                color = map_color
            else:
                color = bar_colors[::-1]

            # Find the indices of the top 20 values
            sorted_indices = np.argsort(count_values)[::-1][:50]

            # Plot the data
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 5))

            # Plot all regions with their corresponding counts and colors
            bars = ax.bar(regions[::-1], count_values[::-1], color=color, width=0.8)

            # Add a black horizontal line representing the mean of count_values
            ax.axhline(mean_counts, color='black', linestyle='--', linewidth=1)

            # Flip the x-axis to make the bars grow leftwards
            ax.invert_xaxis()

            # Set labels and layout
            ax.set_xlabel('Brain regions')
            ax.set_xticks([])
            # ax.set_ylabel('Mean Voxel Intensity')
            ax.set_ylabel('CPM per region')

            # Only label the top 20 regions with their names
            for idx in sorted_indices:
                bar = bars[::-1][idx]  # Reverse the order to match the plot
                ax.text(bar.get_x() + bar.get_width(), bar.get_height(), f'{regions[idx]}',
                        ha='left', va='center', fontsize=10)


            # Adjust the layout
            fig.tight_layout()

            # Save the plot
            plt.savefig(os.path.join(saving_directory, f"{map_name}{dtype}_per_region_top20_with_labels.png"), dpi=300)
            plt.savefig(os.path.join(saving_directory, f"{map_name}{dtype}_per_region_top20_with_labels.svg"), dpi=300)

            plt.show()

            # ########################################################################################################################
            # # Bar plot: density all brain sorted
            # ########################################################################################################################
            #
            # if map_color:
            #     color = map_color
            # else:
            #     color = sorted_bar_colors
            #
            # # Plot the data
            # fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(25, 3))
            #
            # # Plot regions
            # ax.bar(sorted_regions, sorted_count_values, color=sorted_bar_colors, width=0.8)
            # ax.set_xlabel('Intensity per region')
            # ax.invert_xaxis()  # Flip the x-axis to make the bars grow leftwards
            #
            # # Add a black horizontal line representing the mean of density_values
            #
            # ax.axhline(mean_counts, color='black', linestyle='--', linewidth=1)
            #
            # # Set the y-range
            # # ax.set_ylim(0, 0.025)
            # # ax.set_ylim(0, 0.1)
            #
            # # Align the y-axis labels
            # # fig.tight_layout()
            # plt.savefig(os.path.join(saving_directory,
            #                          f"{map_name}_per_region_sorted.png"), dpi=300)
            # plt.savefig(os.path.join(saving_directory,
            #                          f"{map_name}_per_region_sorted.svg"), dpi=300)
            # plt.show()
