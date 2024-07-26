import os

import numpy as np
import pandas as pd
import cv2
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import tifffile


def remove_spines_and_ticks(ax):
    # Turn off the axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Turn off the ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def setup_plot(n, i):
    if i is None and n == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        return fig, ax
    elif i == 0 and n == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        return fig, ax


def plot_cells(filtered_points, reference, tissue_mask, neuronal_mask=None, cell_colors="black", xlim=0, ylim=0, orix=0,
               oriy=1, orip=0, mask_axis=0, s=0.5, saving_path=""):
    if filtered_points.size > 0:
        if neuronal_mask is None:
            filtered_points_plot_x = filtered_points[:, orix]
            filtered_points_plot_y = filtered_points[:, oriy]
        else:
            filtered_points_plot_x = filtered_points[:, orix][~neuronal_mask]
            filtered_points_plot_y = filtered_points[:, oriy][~neuronal_mask]
            if type(cell_colors) != str and cell_colors.size > 0:
                cell_colors = cell_colors[~neuronal_mask]

        max_proj_reference = np.rot90(np.max(reference, axis=orip))[::-1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(filtered_points_plot_x, filtered_points_plot_y, c=cell_colors, s=s,
                   lw=0, edgecolors="black", alpha=1)
        ax.imshow(max_proj_reference, cmap='gray_r', alpha=0.3)
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        ax.invert_yaxis()
        ax.axis('off')
        fig.savefig(saving_path, dpi=300)
        plt.close(fig)

        # Zoom-in view
        split_path = saving_path.split(".")
        max_proj_mask = np.max(tissue_mask, mask_axis)
        if mask_axis == 2:
            max_proj_mask = np.swapaxes(max_proj_mask, 0, 1)

        tifffile.imwrite(split_path[0] + "_test." + split_path[-1], max_proj_mask)
        top_left, bottom_right = find_square_bounding_box(max_proj_mask, 15)
        cropped_ref = max_proj_reference[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        # Adjust the filtered points to the new coordinate system
        adjusted_points_plot_x = filtered_points_plot_x - top_left[1]
        adjusted_points_plot_y = filtered_points_plot_y - top_left[0]

        # Calculate the area of the crop
        crop_height = bottom_right[0] - top_left[0]
        crop_width = bottom_right[1] - top_left[1]
        crop_area = crop_height * crop_width

        # Adjust the size of the points based on the crop area
        adjusted_size = s * (1e5 / crop_area)  # Adjust this scaling factor as needed

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(adjusted_points_plot_x, adjusted_points_plot_y, c=cell_colors, s=adjusted_size,
                   lw=0, edgecolors="black", alpha=1)
        ax.imshow(max_proj_reference[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], cmap='gray_r',
                  alpha=0.3)
        ax.set_xlim(0, cropped_ref.shape[0])
        ax.set_ylim(0, cropped_ref.shape[-1])
        ax.invert_yaxis()
        ax.axis('off')
        fig.savefig(split_path[0] + "_zoom." + split_path[-1], dpi=300)
        plt.close(fig)

        # Contour plot for each color
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(cropped_ref, cmap='gray_r', alpha=0.3)
        ax.set_xlim(0, cropped_ref.shape[1])
        ax.set_ylim(0, cropped_ref.shape[0])
        ax.invert_yaxis()
        ax.axis('off')

        # Find the top 3 most abundant colors
        color_counts = Counter(cell_colors)
        top_colors = [color for color, count in color_counts.most_common(3)]

        # Create contour plots for each of the top 3 colors
        for color in top_colors:
            color_mask = np.array(cell_colors) == color
            x = adjusted_points_plot_x[color_mask]
            y = adjusted_points_plot_y[color_mask]
            if len(x) > 0 and len(y) > 0:
                # Create a density map
                heatmap, xedges, yedges = np.histogram2d(y, x, bins=[np.arange(0, cropped_ref.shape[0]),
                                                                     np.arange(0, cropped_ref.shape[1])])
                heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
                ax.contour(heatmap, levels=5, colors=color, linewidths=1)

        contour_saving_path = split_path[0] + "_contour." + split_path[-1]
        fig.savefig(contour_saving_path, dpi=300)
        plt.close(fig)  # Close the figure to free up memory


def find_square_bounding_box(mask, padding=10):
    # Find all non-zero points in the mask
    points = np.argwhere(mask > 0)

    # Get the bounding box of the non-zero points
    top_left = points.min(axis=0)
    bottom_right = points.max(axis=0)

    # Calculate the width and height of the bounding box
    height = bottom_right[0] - top_left[0]
    width = bottom_right[1] - top_left[1]

    # Determine the side length of the square box
    side_length = max(height, width) + 2 * padding

    # Calculate the center of the bounding box
    center = (top_left + bottom_right) / 2

    # Calculate the top-left and bottom-right corners of the square box
    half_side_length = side_length // 2
    top_left_square = (center - half_side_length).astype(int)
    bottom_right_square = (center + half_side_length).astype(int)

    # Ensure the coordinates are within the image boundaries
    top_left_square = np.maximum(top_left_square, 0)
    bottom_right_square = np.minimum(bottom_right_square, [mask.shape[0], mask.shape[1]])

    return top_left_square, bottom_right_square


def stacked_bar_plot_atlas_regions(sorted_region_id_counts, region_id_counts_total, acros, colors, saving_path):
    # Extract the keys and values
    ids = list(sorted_region_id_counts.keys())
    counts = list(sorted_region_id_counts.values())

    # Extract the acronyms and colors in the same sorted order
    sorted_acros = [acros[ids.index(id)] for id in sorted_region_id_counts.keys()]
    sorted_colors = [colors[ids.index(id)] for id in sorted_region_id_counts.keys()]

    # Reverse the order to have largest counts on top
    reversed_counts = counts[::-1]
    reversed_acros = sorted_acros[::-1]
    reversed_colors = sorted_colors[::-1]

    fraction_reversed_counts = np.array(counts[::-1]) / np.array(region_id_counts_total[::-1]) * 100

    ####################################################################################################################
    # NUMBER OF VOXELS FROM THE BLOB IN REGION
    ####################################################################################################################

    fig, ax = plt.subplots(figsize=(3, 8))  # Create a vertical stacked bar plot
    # Plot a single vertical bar with the counts stacked, largest counts on top
    bars = ax.bar(0, reversed_counts, bottom=np.cumsum([0] + reversed_counts[:-1]), color=reversed_colors,
                  edgecolor='black', linewidth=0.8)
    # Add text labels to the segments
    for i, (count, label) in enumerate(zip(reversed_counts, reversed_acros)):
        ax.text(0.5, np.cumsum([0] + reversed_counts[:i + 1])[-1] - count / 2, str(label), ha='left', va='center',
                fontsize=10, color="black")
    ax.set_ylabel('Voxels from region in blob')  # Set y-axis label
    ax.xaxis.set_visible(False)  # Remove x-axis labels and ticks
    # Remove the top, right, and left spines (the square around the plot)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    plt.tight_layout()  # Show the plot
    plt.savefig(os.path.join(saving_path, "voxels_from_region_in_blob.png"), dpi=300)
    plt.savefig(os.path.join(saving_path, "voxels_from_region_in_blob.svg"), dpi=300)

    ####################################################################################################################
    # FRACTION OF THE ENTIRE REGION
    ####################################################################################################################

    sorted_indices = np.argsort(fraction_reversed_counts)[::-1]
    sorted_acros = np.array(reversed_acros)[sorted_indices]
    sorted_counts = np.array(fraction_reversed_counts)[sorted_indices]
    sorted_colors = np.array(reversed_colors)[sorted_indices]

    n_bars = 10
    x_axis = np.arange(0, n_bars, 1)
    if len(sorted_acros) < n_bars:
        sorted_acros = list(sorted_acros) + [None] * (n_bars - len(sorted_acros))
    #     sorted_counts = list(sorted_counts) + [None] * (n_bars - len(sorted_counts))
    #     sorted_colors = list(sorted_colors) + [None] * (n_bars - len(sorted_colors))
    if len(sorted_acros) > n_bars:
        sorted_acros = sorted_acros[:n_bars]
        sorted_counts = sorted_counts[:n_bars]
        sorted_colors = sorted_colors[:n_bars]

    fig, ax = plt.subplots(figsize=(3, 8))  # Create a vertical stacked bar plot
    # Plot a single vertical bar with the counts stacked, largest counts on top
    bars = ax.bar(np.arange(0, len(sorted_counts), 1), sorted_counts,
                  color=sorted_colors,
                  edgecolor='black', linewidth=0.8)
    # Add text labels to the segments
    ax.set_ylabel('Fraction of voxels from region in blob (%)')  # Set y-axis label
    ax.set_xticks(x_axis)
    ax.set_xticklabels(sorted_acros, fontsize=10, rotation=90)
    # Remove the top, right, and left spines (the square around the plot)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()  # Show the plot
    plt.savefig(os.path.join(saving_path, "fraction_voxels_from_region_in_blob.png"), dpi=300)
    plt.savefig(os.path.join(saving_path, "fraction_voxels_from_region_in_blob.svg"), dpi=300)


def bar_plot(cells_color, unique_cells, unique_cells_color, saving_path, n_bars=20):
    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 10))

    # Count the occurrences of each color in labels
    color_counts = pd.Series(np.array(cells_color)).value_counts() \
        .reindex(unique_cells_color, fill_value=0)
    # Create a DataFrame for sorting
    count_df = pd.DataFrame({
        'Color': unique_cells_color,
        'Label': unique_cells,
        'Count': color_counts
    })

    # Sort the DataFrame by counts in descending order and select top 10
    sorted_df = count_df.sort_values(by='Count', ascending=False).head(n_bars)

    # Bar plot on the second subplot for top 20 items
    ax1.bar(range(len(sorted_df)), sorted_df['Count'], color=sorted_df['Color'], width=0.8, lw=0.8, edgecolor="black")
    ax1.set_ylabel('Number of Cells', fontsize=12)
    ax1.set_xlim(-0.5, n_bars-0.5)
    ax1.set_xticks([])

    top_handles = [mlines.Line2D([0], [0], color=c, marker='o', linestyle='None', markersize=10)
                   for c in sorted_df['Color']]
    top_labels = sorted_df['Label'].tolist()
    ax1.legend(handles=top_handles, labels=top_labels, fontsize=10, ncol=1, loc=1)

    plt.tight_layout()
    plt.savefig(saving_path, dpi=300)


def stacked_horizontal_bar_plot(categories, data_categories, saving_path, labeled=False):
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(len(data_categories), 1, hspace=0.1)  # Adjust hspace for better spacing

    previous_right_ends = None

    for n, (cat, data) in enumerate(zip(categories, data_categories)):
        ax = fig.add_subplot(gs[n])

        left = 0
        current_right_ends = []
        for index, row in data.iterrows():
            bar = ax.barh(0, row['Count'], left=left, color=row['Color'], edgecolor="black", lw=0.5, zorder=1)
            current_right_ends.append(left + row['Count'])

            if labeled:
                ax.text(left + row['Count'] / 2, 0, str(row['Label']) + "\n" + str(row['Count']),
                        ha='center', va='center', fontsize=4, zorder=3,
                        rotation=90)

            left += row['Count']

        if previous_right_ends is not None:
            for prev_right_end, curr_right_end in zip(previous_right_ends, current_right_ends):
                ax.plot([prev_right_end, prev_right_end], [1, 0], color='black', lw=0.5, clip_on=False, zorder=2)

        previous_right_ends = current_right_ends

        if n + 1 == len(data_categories):
            ax.set_xlabel('Number of Cells', fontsize=10)
        else:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)

        ax.set_ylabel(cat, fontsize=10)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlim(0, sum(data['Count']))
        ax.set_ylim(-0.5, 0.5)

    plt.tight_layout()
    plt.savefig(saving_path + ".png", dpi=300)
    plt.savefig(saving_path + ".svg", dpi=300)
