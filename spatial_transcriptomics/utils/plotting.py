import os

import numpy as np
import matplotlib.pyplot as plt

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
