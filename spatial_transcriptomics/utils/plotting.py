import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


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


def plot_cells(n, i, fig, ax, cell_colors="black", neuronal_mask=None, xlim=0, ylim=0, orix=0, oriy=1, orip=0,
               saving_name="", **kwargs):
    if kwargs["filtered_points"].size > 0:
        if neuronal_mask is None:
            filtered_points_plot_x = kwargs["filtered_points"][:, orix]
            filtered_points_plot_y = kwargs["filtered_points"][:, oriy]
        else:
            filtered_points_plot_x = kwargs["filtered_points"][:, orix][~neuronal_mask]
            filtered_points_plot_y = kwargs["filtered_points"][:, oriy][~neuronal_mask]
            if type(cell_colors) != str and cell_colors.size > 0:
                cell_colors = cell_colors[~neuronal_mask]

        ax.scatter(filtered_points_plot_x, filtered_points_plot_y, c=cell_colors, s=kwargs["marker_size"],
                   lw=kwargs["linewidth"], edgecolors="black", alpha=1)

    if i is None and n + 1 == kwargs["n_chunks"]:
        ax.imshow(np.rot90(np.max(kwargs["reference"], axis=orip))[::-1], cmap='gray_r', alpha=0.3)
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        ax.invert_yaxis()
        ax.axis('off')
        fig.savefig(os.path.join(kwargs["saving_dir"], saving_name), dpi=300)
    elif i is not None and i + 1 == kwargs["n_datasets"] and n + 1 == kwargs["n_chunks"]:
        ax.imshow(np.rot90(np.max(kwargs["reference"], axis=orip))[::-1], cmap='gray_r', alpha=0.3)
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        ax.invert_yaxis()
        ax.axis('off')
        fig.savefig(os.path.join(kwargs["saving_dir"], saving_name), dpi=300)


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
    ax.spines['left'].set_visible(False)
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
