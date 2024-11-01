import os
import json

import numpy as np
import pandas as pd
# import cv2
import scipy.ndimage
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import to_tree
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import tifffile

import utils.utils as ut

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


def plot_cells(filtered_points, reference, tissue_mask, non_neuronal_mask=None, cell_colors="black", cell_categories=None,
               xlim=0, ylim=0, orix=0, oriy=1, orip=0, ori="", mask_axis=0, s=0.5, sg=0.5, saving_path="", relevant_categories=[],
               show_outline=False, zoom=False, plot_individual_categories=True, show_ref=True, surface_projection=True,
               show_cluster_name=False):

    # If at least one set of coordinates for a cell is given
    if filtered_points.size > 0:

        if cell_categories is None:
            cell_categories = np.array([])

        ################################################################################################################
        #  FILTER THE CELLS TO BE KEPT (ALL CELLS / NEURONS) AND FETCH
        ################################################################################################################

        ut.print_c("[INFO] Filtering cells to be kept and fetches them!")

        if non_neuronal_mask is None:
            filtered_points_plot_x = filtered_points[:, orix]
            filtered_points_plot_y = filtered_points[:, oriy]
            filtered_points_plot_z = filtered_points[:, mask_axis]
            if type(cell_colors) == str:
                cell_colors = np.full(filtered_points.shape[:1], cell_colors)
        else:
            filtered_points_plot_x = filtered_points[:, orix][~non_neuronal_mask]
            filtered_points_plot_y = filtered_points[:, oriy][~non_neuronal_mask]
            filtered_points_plot_z = filtered_points[:, mask_axis][~non_neuronal_mask]
            if type(cell_colors) != str and cell_colors.size > 0:
                cell_colors = cell_colors[~non_neuronal_mask]
            else:
                cell_colors = np.full(filtered_points.shape[:1], cell_colors)[~non_neuronal_mask]
            if cell_categories.size > 0:
                cell_categories = cell_categories[~non_neuronal_mask]

        # category_counts = Counter(cell_categories)
        # sorted_category_counts = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))
        # categories = list(sorted_category_counts.keys())
        categories = np.unique(cell_categories)
        # categories = np.array(categories)
        n_categories = len(categories)
        if cell_colors.size > 0:
            colors = []
            for u, ucat in enumerate(categories):
                ut.print_c(f"[INFO] Fetching color for category: {ucat}; {u+1}/{n_categories}")
                colors.append(cell_colors[cell_categories == ucat][0])
            colors = np.array(colors)
        else:
            colors = np.array([])

        # color_counts = Counter(cell_colors)
        # sorted_color_counts = dict(sorted(color_counts.items(), key=lambda item: item[1], reverse=True))
        # colors = list(sorted_color_counts.keys())
        # # colors = np.array([cell_colors[cell_categories == i][0] for i in cell_categories])
        # if cell_categories.size > 0:
        #     # categories = np.array([cell_categories[cell_colors == i][0] for i in colors])
        #     categories = cell_categories
        # else:
        #     categories = np.array([])

        # If we keep only specific categories
        if relevant_categories:
            category_masks = np.array([True if i in relevant_categories else False for i in categories])
            categories = categories[category_masks]
            colors = np.array(colors)[category_masks]

        ################################################################################################################
        #  GENERATE THE REFERENCE OVERLAY IMAGE
        ################################################################################################################

        # DETERMINE CROP SIZE

        max_proj_reference = np.rot90(np.max(reference, axis=orip))[::-1]
        ref_alpha = 0.2
        max_proj_mask = np.max(tissue_mask, mask_axis)
        if mask_axis == 2:
            max_proj_mask = np.swapaxes(max_proj_mask, 0, 1)

        top_left, bottom_right = find_square_bounding_box(max_proj_mask, 15)
        cropped_ref = max_proj_reference[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        saving_file_name = os.path.basename(saving_path)
        split_path = saving_file_name.split(".")

        if not os.path.exists(os.path.join(os.path.dirname(saving_path), f"mask_{ori}.png")):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(max_proj_mask, cmap='gray_r',
                      alpha=0.3)
            ax.set_xlim(0, max_proj_reference.shape[1])
            ax.set_ylim(0, max_proj_reference.shape[0])
            ax.invert_yaxis()
            ax.axis('off')
            fig.savefig(os.path.join(os.path.dirname(saving_path), f"mask_{ori}.png"), dpi=300)
            fig.savefig(os.path.join(os.path.dirname(saving_path), f"mask_{ori}.svg"), dpi=300)
            plt.close(fig)

        if not os.path.exists(os.path.join(os.path.dirname(saving_path), f"mask_{ori}_zoom.png")):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(max_proj_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], cmap='gray_r',
                      alpha=0.3)
            ax.set_xlim(0, cropped_ref.shape[1])
            ax.set_ylim(0, cropped_ref.shape[0])
            ax.invert_yaxis()
            ax.axis('off')
            fig.savefig(os.path.join(os.path.dirname(saving_path), f"mask_{ori}_zoom.png"), dpi=300)
            fig.savefig(os.path.join(os.path.dirname(saving_path), f"mask_{ori}_zoom.svg"), dpi=300)
            plt.close(fig)


        if categories.size > 0 and plot_individual_categories:

            for color, category in zip(colors, categories):

                ut.print_c(f"[INFO] Plotting category: {category}!")
                category_name_corrected = category.replace("/", "-")
                saving_dir = ut.create_dir(os.path.join(os.path.dirname(saving_path), category_name_corrected))
                #color_mask = np.array(cell_colors) == color
                cat_mask = np.array(cell_categories) == category

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(filtered_points_plot_x[cat_mask], filtered_points_plot_y[cat_mask], c=color, s=sg,
                           lw=0, edgecolors="black", alpha=1)
                if show_outline:
                    ax.contour(max_proj_mask, levels=[0.5], colors='black', linewidths=1, alpha=0.5, linestyles='dashed')
                if show_ref:
                    ax.imshow(max_proj_reference, cmap='gray_r', alpha=ref_alpha)
                else:
                    ax.imshow(max_proj_reference,
                              cmap='gray_r',
                              alpha=0)
                ax.set_xlim(0, xlim)
                ax.set_ylim(0, ylim)
                ax.invert_yaxis()
                ax.axis('off')
                fig.savefig(os.path.join(saving_dir, saving_file_name), dpi=300)
                plt.close(fig)

                if zoom:

                    # Zoom-in view
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
                    ax.scatter(adjusted_points_plot_x[cat_mask], adjusted_points_plot_y[cat_mask], c=color, s=adjusted_size,
                               lw=0, edgecolors="black", alpha=1)
                    if show_ref:
                        ax.imshow(max_proj_reference[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], cmap='gray_r',
                                  alpha=ref_alpha)
                    else:
                        ax.imshow(max_proj_reference[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], cmap='gray_r',
                                  alpha=0)
                    if show_outline:
                        ax.contour(max_proj_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], levels=[0.5],
                                   colors='black', linewidths=1, alpha=0.5, linestyles='dashed')
                    ax.set_xlim(0, cropped_ref.shape[1])
                    ax.set_ylim(0, cropped_ref.shape[0])
                    ax.invert_yaxis()
                    ax.axis('off')
                    fig.savefig(os.path.join(saving_dir, split_path[0] + "_zoom." + split_path[-1]), dpi=300)
                    fig.savefig(os.path.join(saving_dir, split_path[0] + "_zoom.svg"), dpi=300)
                    plt.close(fig)

                    ############################################################################################################
                    # CONTOUR PLOT
                    ############################################################################################################

                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    # Display the reference image
                    if show_ref:
                        ax.imshow(cropped_ref, cmap='gray_r', alpha=ref_alpha)
                    else:
                        ax.imshow(cropped_ref, cmap='gray_r', alpha=0)
                    ax.set_xlim(0, cropped_ref.shape[1])
                    ax.set_ylim(0, cropped_ref.shape[0])
                    ax.invert_yaxis()
                    ax.axis('off')

                    # Create contour plots for each of the colors
                    x = adjusted_points_plot_x[cat_mask]
                    y = adjusted_points_plot_y[cat_mask]
                    y_flipped = cropped_ref.shape[0] - adjusted_points_plot_y[cat_mask]

                    if len(x) > 0 and len(y) > 0:
                        # Create a higher resolution density map
                        x_edges = np.linspace(0, cropped_ref.shape[1], cropped_ref.shape[1] * 2 + 1)
                        y_edges = np.linspace(0, cropped_ref.shape[0], cropped_ref.shape[0] * 2 + 1)

                        # Note the order of bins: numpy's histogram2d takes [y_edges, x_edges]
                        heatmap, xedges, yedges = np.histogram2d(y_flipped, x, bins=[y_edges, x_edges])

                        # Apply Gaussian smoothing
                        sigma = 1.5  # Higher sigma for more smoothing
                        smoothed_heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=sigma)

                        # Adjust the levels to avoid emphasizing low-density regions
                        min_level = np.percentile(smoothed_heatmap, 20)  # Exclude lower percentiles
                        max_level = np.max(smoothed_heatmap)
                        levels = np.linspace(min_level, max_level, 6)

                        # Correct the extent to match the image orientation
                        extent = [0, cropped_ref.shape[1], cropped_ref.shape[0], 0]

                        # Calculate alpha values for each level, more transparent on outer levels
                        alphas = np.linspace(0.1, 1.0,
                                             len(levels) - 1)  # Transparency gradient from outer (0.1) to inner (1.0)

                        for i in range(len(levels) - 1):
                            ax.contour(smoothed_heatmap, levels=[levels[i], levels[i + 1]], colors=[color], alpha=alphas[i],
                                       linewidths=1, extent=extent)

                    if show_outline:
                        ax.contour(max_proj_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], levels=[0.5],
                                   colors='black', linewidths=1, alpha=0.5, linestyles='dashed')

                    contour_saving_path = split_path[0] + "_contour." + split_path[-1]
                    fig.savefig(os.path.join(saving_dir, contour_saving_path), dpi=300)
                    fig.savefig(os.path.join(saving_dir, split_path[0] + "_contour.svg"), dpi=300)
                    plt.close(fig)  # Close the figure to free up memory

        if ori == "sagittal":
            views = range(1)
        else:
            views = range(2)

        for view in views:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            if surface_projection:
                sorted_z_indices = np.argsort(filtered_points_plot_z)
                if view == 0:
                    sorted_z_indices = sorted_z_indices[::-1]
                # Apply this sorting to the x, y, and z coordinates
                sorted_x_z = filtered_points_plot_x[sorted_z_indices]
                sorted_y_z = filtered_points_plot_y[sorted_z_indices]
                sorted_colors_z = cell_colors[sorted_z_indices]
                sorted_categories = cell_categories[sorted_z_indices]
            else:
                sorted_x_z = filtered_points_plot_x
                sorted_y_z = filtered_points_plot_y
                sorted_colors_z = cell_colors
                sorted_categories = cell_categories

            ax.scatter(sorted_x_z, sorted_y_z, c=sorted_colors_z, s=sg,
                       lw=0, edgecolors="black", alpha=1)

            # Add cluster names
            if show_cluster_name:
                for cluster_name, center_x, center_y, text_color in zip(sorted_categories, sorted_x_z, sorted_y_z,
                                                                        sorted_colors_z):
                    ax.text(center_x, center_y, cluster_name, fontsize=3, ha='center', va='center', color=text_color)

            if show_outline:
                ax.contour(max_proj_mask, levels=[0.5], colors='black', linewidths=1, alpha=0.5, linestyles='dashed')
            if show_ref:
                ax.imshow(max_proj_reference, cmap='gray_r', alpha=0.3)
            else:
                ax.imshow(max_proj_reference, cmap='gray_r', alpha=0)
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)
            ax.invert_yaxis()
            ax.axis('off')
            fig.savefig(os.path.join(os.path.dirname(saving_path), split_path[0] + f"_{view}." + split_path[-1]),
                        dpi=300)
            plt.close(fig)

            if zoom:

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
                if show_ref:
                    ax.imshow(max_proj_reference[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], cmap='gray_r',
                              alpha=ref_alpha)
                else:
                    ax.imshow(max_proj_reference[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]],
                              cmap='gray_r',
                              alpha=0)
                if show_outline:
                    ax.contour(max_proj_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]], levels=[0.5],
                               colors='black', linewidths=1, alpha=0.5, linestyles='dashed')
                ax.set_xlim(0, cropped_ref.shape[0])
                ax.set_ylim(0, cropped_ref.shape[-1])
                ax.invert_yaxis()
                ax.axis('off')
                fig.savefig(os.path.join(os.path.dirname(saving_path), split_path[0] + f"_{view}_zoom." + split_path[-1]),
                            dpi=300)
                plt.close(fig)


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
    fraction_reversed_counts = [i if i <= 100 else 100 for i in fraction_reversed_counts]  # Fixme: Check why some region are above 100

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


def stacked_horizontal_bar_plot(categories, data_categories, saving_path, plots_to_generate=["categories"],
                                colormap="viridis", max_range=100):
    saving_directory = saving_path
    cmap = plt.get_cmap(colormap)  # You can choose different colormaps like 'plasma', 'inferno', etc.
    lw = 0.05

    for plot in plots_to_generate:

        previous_right_ends = None
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(len(data_categories), 2, width_ratios=[0.95, 0.05], wspace=0.05)  # Add space for colorbar

        for n, (cat, data) in enumerate(zip(categories, data_categories)):

            # Sorting the dataframe by the numeric part of the 'Label' column
            if cat == "neurotransmitter":
                data['Label_Number'] = data['Label_cluster'].str.extract(r'(\d+)').astype(int)
            else:
                data['Label_Number'] = data['Label'].str.extract(r'(\d+)').astype(int)
            data = data.sort_values(by='Label_Number').drop(columns=['Label_Number'])

            ax = fig.add_subplot(gs[n, 0])

            left = 0
            current_right_ends = []
            for index, row in data.iterrows():
                # Calculate the width for the full opacity part
                if plot.startswith("categories"):
                    full_opacity_width = row['Count_df']
                    # Draw the full opacity part
                    bar_full_opacity = ax.barh(0, full_opacity_width, left=left, color=row['Color_df'],
                                               edgecolor="black", lw=lw, zorder=1)
                    left += full_opacity_width
                elif plot.startswith("percentage"):
                    full_opacity_width = row['Count_df']
                    percentage = row['Percentage']
                    print(percentage)
                    if percentage > max_range:
                        percentage = max_range
                    normalized = float(percentage / max_range)
                    colors = cmap(normalized)
                    bar_full_opacity = ax.barh(0, full_opacity_width, left=left, color=colors,  #row['Color_df']
                                               edgecolor="black", lw=lw, zorder=1)
                    left += full_opacity_width
                current_right_ends.append(left)

                if plot.endswith("labeled"):
                    if cat != "neurotransmitter":
                        ax.text(left - (row['Count_df'] / 2), 0, str(row['Label']) + "\n" + str(row['Count_df']),
                                ha='center', va='center', fontsize=4, zorder=3,
                                rotation=90)
                    else:
                        ax.text(left - (row['Count_df'] / 2), 0, str(row['Label_cluster']) + "\n" + str(row['Count_df']),
                                ha='center', va='center', fontsize=4, zorder=3,
                                rotation=90)

            if previous_right_ends is not None:
                for prev_right_end, curr_right_end in zip(previous_right_ends, current_right_ends):
                    ax.plot([prev_right_end, prev_right_end], [0.8, 0.4], color='black', lw=0.1, clip_on=False, zorder=2)

            previous_right_ends = current_right_ends

            if n + 1 == len(data_categories):
                ax.set_xlabel('Number of Cells', fontsize=10)
            else:
                ax.set_xticks([])
                ax.spines['bottom'].set_visible(False)

            ax.set_ylabel(cat + ": " + str(len(data)), fontsize=10)
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlim(0, sum(data['Count_df']))
            ax.set_ylim(-0.5, 0.5)

        # Add a single colorbar to the entire figure
        if plot.startswith("percentage"):
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_range))
            sm.set_array([])
            cbar_ax = fig.add_subplot(gs[:, 1])  # Allocate the colorbar area on the right
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
            cbar.set_label('Percentage')

        # plt.tight_layout()

        if plot.startswith("categories"):
            saving_path = os.path.join(saving_path, "categories")
        if plot.startswith("percentage"):
            saving_path = os.path.join(saving_path, "percentage")
        if plot.endswith("labeled"):
            saving_path = saving_path + "_labeled"

        plt.savefig(saving_path + ".png", dpi=300)
        plt.savefig(saving_path + ".svg", dpi=300)
        saving_path = saving_directory
        plt.close()


def create_ordered_dendrogram(categories, data_categories, saving_path):
    hierarchy = []
    node_count = 0  # Unique node count for creating the linkage matrix
    node_dict = {}  # Map each label to a unique node index

    def group_by_cumulative_sum(parent_count, child_df):
        """
        Groups child nodes based on a cumulative sum of counts that does not exceed the parent count.
        """
        grouped_indexes = []
        current_sum = 0

        for idx, row in child_df.iterrows():
            grouped_indexes.append(idx)
            current_sum += row['Count_df']

            if parent_count is not None and current_sum >= parent_count:
                break

        return grouped_indexes

    def traverse_hierarchy(level, parent_indices=None, parent_label=None):
        """
        Recursively traverses the hierarchy to build the ordered dendrogram structure.
        Processes all supertypes and clusters at the current level before moving to the next level.
        """
        if level >= len(categories):
            return

        parent_df = hierarchy[level - 1] if level > 0 else None
        child_df = data_categories[level].copy()

        # Initialize parent_indices if not provided
        if parent_indices is None:
            parent_indices = list(range(len(child_df)))  # Start with all indices at the first level

        labels, counts, percentages, colors, parents = [], [], [], [], []

        print(parent_indices)

        # Step 1: Process each parent cluster at the current level
        for parent_index in parent_indices:
            # Determine grouping based on 'Count_df'
            if parent_df is not None:
                if parent_index >= len(parent_df['counts']):
                    print(f"Warning: parent_index {parent_index} is out of range for level {level}. Skipping.")
                    continue  # Skip if parent_index is invalid

                parent_count = parent_df['counts'][parent_index]
                child_indexes = group_by_cumulative_sum(parent_count, child_df)
                grouped_child_df = child_df.loc[child_indexes]
                print(parent_df, parent_count)
                print(child_df, child_indexes)
                print(grouped_child_df)
            else:
                # First level sorted by 'Count_df'
                grouped_child_df = child_df.sort_values(by='Count_df', ascending=False)

            # Collect details of the current grouped child data frame
            for idx in grouped_child_df.index:
                labels.append(grouped_child_df.loc[idx, 'Label'])
                counts.append(grouped_child_df.loc[idx, 'Count_df'])
                percentages.append(grouped_child_df.loc[idx, 'Percentage'])
                colors.append(grouped_child_df.loc[idx, 'Color_df'])
                parents.append(parent_label if parent_label else None)

        # Add the grouped data for this level to the hierarchy
        hierarchy.append({
            'labels': labels,
            'counts': counts,
            'percentages': percentages,
            'colors': colors,
            'parent': parents
        })

        # Step 2: Recursively process the next level for each unique parent_label (subtype) in this level
        if len(labels) > 0:
            processed_labels = set()  # Track processed parent labels to avoid duplication
            for idx, label in enumerate(labels):
                if label not in processed_labels:
                    processed_labels.add(label)

                    # Find child indices associated with this parent label
                    child_parent_indices = [i for i, parent in enumerate(parents) if parent == label]

                    # Recursively process the next level (subtypes) for each unique parent label
                    traverse_hierarchy(level + 1, child_parent_indices, label)

    # Initialize the hierarchy traversal
    traverse_hierarchy(0)

    # Save the hierarchy structure to a file for inspection
    hierarchy_path = os.path.join(saving_path, "hierarchy_structure.json")
    with open(hierarchy_path, "w") as f:
        json.dump(hierarchy, f, indent=4)

    print(f"Hierarchy structure saved to {hierarchy_path} for inspection.")

    # Prepare the linkage matrix for the dendrogram
    Z = []
    labels_at_leaves = []

    for level, h in enumerate(hierarchy):
        for idx, label in enumerate(h['labels']):
            if level == 0:
                # The first level should only initialize labels and nodes
                node_name = f"{categories[level]}_{label}"
                node_dict[node_name] = node_count
                labels_at_leaves.append(label)
                node_count += 1
                continue

            # Ensure the level index is within bounds
            if level >= len(categories):
                print(f"Warning: Level index {level} is out of range for categories.")
                continue

            node_name = f"{categories[level]}_{label}"
            node_dict[node_name] = node_count
            labels_at_leaves.append(label)
            node_count += 1

            if h['parent'][idx] is not None:
                parent_name = f"{categories[level - 1]}_{h['parent'][idx]}"
                if parent_name in node_dict:
                    parent_node = node_dict[parent_name]
                    Z.append([parent_node, node_dict[node_name], h['percentages'][idx], 1])
                else:
                    print(f"Warning: Parent {parent_name} not found for label {label} at level {level}.")

    Z = np.array(Z)

    # Ensure the number of leaves in Z matches the labels
    if Z.shape[0] != len(labels_at_leaves) - 1:
        print(f"Mismatch in the number of leaves and labels in the dendrogram.")
        print(f"Expected: {Z.shape[0] + 1}, but got {len(labels_at_leaves)}.")
        return

    # Plotting the dendrogram
    plt.figure(figsize=(15, 10))
    dendrogram(Z, labels=labels_at_leaves, orientation='left', color_threshold=0, above_threshold_color='black')
    plt.savefig(os.path.join(saving_path, "dendrogram.png"), dpi=300)
    plt.savefig(os.path.join(saving_path, "dendrogram.svg"), dpi=300)
    plt.close()

    print(f"Dendrogram saved to {saving_path}.")
