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


def bar_plot(cells_color, unique_cells, unique_cells_color, saving_path):
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
    sorted_df = count_df.sort_values(by='Count', ascending=False).head(10)

    # Bar plot on the second subplot for top 20 items
    ax1.bar(range(len(sorted_df)), sorted_df['Count'], color=sorted_df['Color'])
    ax1.set_ylabel('Number of Cells', fontsize=20)
    ax1.set_xlim(-0.5, 10)
    # ax1.set_xticks(range(len(sorted_df)))
    ax1.set_xticks([])
    # ax1.set_xticklabels(sorted_df['Label'], rotation=45, fontsize=15)

    top_10_handles = [mlines.Line2D([0], [0], color=c, marker='o', linestyle='None', markersize=10)
                      for c in sorted_df['Color']]
    top_10_labels = sorted_df['Label'].tolist()
    ax1.legend(handles=top_10_handles, labels=top_10_labels, fontsize=20, ncol=1, loc=1)

    plt.tight_layout()
    plt.savefig(saving_path, dpi=300)
