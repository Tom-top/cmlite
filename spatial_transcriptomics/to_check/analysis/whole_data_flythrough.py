import os

import json
import requests
import numpy as np
import umap
import pandas as pd
import tifffile
import anndata
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import time

matplotlib.use("Agg")

def filter_coordinates_dim(coordinates, i, j, dim):
    """
    Filters out coordinates where the value in a specified dimension falls within the range [i, j]
    and returns a mask indicating which points were kept.

    Parameters:
    coordinates (list of tuples): A list of (x, y, z) coordinates.
    i (float): The lower bound of the value range to exclude.
    j (float): The upper bound of the value range to exclude.
    dim (int): The dimension to check (0 for x, 1 for y, 2 for z).

    Returns:
    tuple: A tuple containing the filtered list of coordinates and a mask.
    """
    mask = np.array([(point[dim] > i and point[dim] < j) for point in coordinates])
    filtered_coordinates = np.array([point for point, m in zip(coordinates, mask) if m])
    return filtered_coordinates, mask


def filter_points_in_3d_mask(arr_0, mask_1):
    start_time = time.time()  # Start the timer

    # Convert coordinates to integers
    int_coords = arr_0.astype(int)

    # Ensure coordinates are within bounds
    valid_x = (0 <= int_coords[:, 0]) & (int_coords[:, 0] < mask_1.shape[0])
    valid_y = (0 <= int_coords[:, 1]) & (int_coords[:, 1] < mask_1.shape[1])
    valid_z = (0 <= int_coords[:, 2]) & (int_coords[:, 2] < mask_1.shape[2])
    in_bounds = valid_x & valid_y & valid_z

    # Create a mask for points within the bounds
    mask_2 = np.zeros(arr_0.shape[0], dtype=bool)
    mask_2[in_bounds] = mask_1[int_coords[in_bounds, 0], int_coords[in_bounds, 1], int_coords[in_bounds, 2]] == 255

    # Filter arr_0 using mask_2
    filtered_arr_0 = arr_0[mask_2]

    end_time = time.time()  # End the timer
    print(f"Function run time: {end_time - start_time} seconds")  # Print the elapsed time

    return filtered_arr_0, mask_2


def umap_hdbscan_clustering(data, return_hdbscan=True):
    """
    Perform UMAP dimensionality reduction followed by HDBSCAN clustering.

    :param data: 2D numpy array of shape (n_samples, n_features).
    :return: UMAP embeddings and HDBSCAN cluster labels.
    """
    # UMAP reduction
    print("Running UMAP reduction!")
    # reducer = umap.UMAP(random_state=42)
    # embedding = reducer.fit_transform(data)
    reducer = umap.UMAP(random_state=999, n_neighbors=30, min_dist=.25)
    embedding = pd.DataFrame(reducer.fit_transform(data), columns=['UMAP1', 'UMAP2'])

    if return_hdbscan:
        # HDBSCAN clustering
        print("Running HDBSCAN clustering!")
        clusterer = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=100)
        cluster_labels = clusterer.fit_predict(embedding)

        return embedding, cluster_labels

    return embedding, None


def plot_umap_with_clusters(embedding, labels, cmap="", title="", legend=[], save=True, alpha=0.5, saving_dir=""):
    # Create a subplot layout: 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [3, 1]})

    # UMAP plot on the first subplot
    if cmap:
        scatter = ax1.scatter(embedding["UMAP1"], embedding["UMAP2"], c=labels, cmap=cmap, s=3, alpha=alpha)
    else:
        scatter = ax1.scatter(embedding["UMAP1"], embedding["UMAP2"], c=labels, s=3, alpha=alpha)
    ax1.set_aspect('equal', 'datalim')
    ax1.set_title('UMAP projection', fontsize=12)
    ax1.set_xlabel('UMAP1')
    ax1.set_ylabel('UMAP2')

    # Count the occurrences of each color in labels
    color_counts = pd.Series(labels).value_counts().reindex(legend[0], fill_value=0)

    # Create a DataFrame for sorting
    count_df = pd.DataFrame({
        'Color': legend[0],
        'Label': legend[1],
        'Count': color_counts
    })

    # Sort the DataFrame by counts in descending order and select top 20
    sorted_df = count_df.sort_values(by='Count', ascending=False).head(20)

    # Update the legend to reflect top 20 items
    if legend:
        top_20_handles = [mlines.Line2D([0], [0], color=c, marker='o', linestyle='None', markersize=10)
                          for c in sorted_df['Color']]
        top_20_labels = sorted_df['Label'].tolist()
        ax1.legend(handles=top_20_handles, labels=top_20_labels, fontsize=8, ncol=1, loc=2)

    # Bar plot on the second subplot for top 20 items
    ax2.bar(range(len(sorted_df)), sorted_df['Count'], color=sorted_df['Color'])
    ax2.set_ylabel('Number of Cells')
    ax2.set_xticks(range(len(sorted_df)))
    ax2.set_xticklabels(sorted_df['Label'], rotation=90, fontsize=8)

    # Adjust the layout
    plt.tight_layout()
    if save:
        plt.savefig(saving_dir, dpi=300)
    # plt.show()


def hex_to_rgb(hex):
    return tuple(int(hex[1:][i:i + 2], 16) for i in (0, 2, 4))


def check_and_create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


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


dataset_id = "Zhuang-ABCA-1"
download_base = r'E:\tto'
url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
manifest = json.loads(requests.get(url).text)
metadata = manifest['file_listing'][dataset_id]['metadata']
metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']
expression_matrices = manifest['file_listing'][dataset_id]['expression_matrices']
cell_metadata_path = expression_matrices[dataset_id]['log2']['files']['h5ad']['relative_path']
file = os.path.join(download_base, cell_metadata_path)
adata = anndata.read_h5ad(file, backed='r')
genes = adata.var

# Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
cell_metadata_file_ccf = os.path.join(download_base, cell_metadata_path_ccf)
cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
cell_metadata_ccf.set_index('cell_label', inplace=True)
cell_labels = cell_metadata_ccf.index
n_cells_ccf = len(cell_metadata_ccf)

# Filter out the cells
transformed_coordinates = np.load(r"E:\tto\results\heatmaps\general\all_transformed_cells.npy")

maps_path = r"U:\Users\TTO\spatial_transcriptomics\neuropedia\maps"
maps = ["all_clusters"]

start, end, step, thickness = 0, 512, 10, 10
thickness_dif = int((thickness - step)/2)
start_steps = np.arange(start, end, step)
end_steps = np.arange(start+step, end+step, step)
end_steps[-1] = end
n_steps = len(start_steps)

for m in maps:

    for p, (s, e) in enumerate(zip(start_steps, end_steps)):

        map_path = os.path.join(maps_path, m)
        if os.path.isdir(map_path):
            res_dir = os.path.join(map_path, fr"transcriptomics\full_data_plots")
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            whole_brain_dir = os.path.join(res_dir, "whole_brain")
            if not os.path.exists(whole_brain_dir):
                os.mkdir(whole_brain_dir)
            mask = tifffile.imread(os.path.join(map_path, r"v5_to_v6\bin.tif"))

            print(f"Generating plots! {p+1}/{n_steps}")
            if s == 0:
                mask[:s] = 0
            else:
                mask[:s - thickness_dif] = 0

            if e == end:
                mask[e:] = 0
            else:
                mask[e + thickness_dif:] = 0

            print("Filtering points in mask!")
            filtered_points, mask_point = filter_points_in_3d_mask(transformed_coordinates, mask)
            filtered_labels = cell_labels[::-1][mask_point]

            adata_filtered = adata[adata.obs.index.isin(filtered_labels), :]
            obs_df = adata_filtered.obs.copy()
            # Reorder obs_df according to filtered_labels
            obs_df = obs_df.loc[filtered_labels]
            # Use the ordered index from obs_df to index into adata
            adata_filtered = adata[obs_df.index, :]
            ts = time.time()
            gdata = adata_filtered.to_df()
            te = time.time()
            print(f"Function run time: {te - ts} seconds")  # Print the elapsed time
            print(f"{len(gdata)} cells found!")

            # Views
            cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
            cell_metadata_file_views = os.path.join(download_base, cell_metadata_path_views)
            cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
            cell_metadata_views.set_index('cell_label', inplace=True)

            # Get the relevant rows from cell_metadata_views in one operation
            print("Filtering cell metadata views!")
            filtered_metadata_views = cell_metadata_views.loc[filtered_labels]

            # Extract data for each category
            cells_class = filtered_metadata_views["class"].tolist()
            cells_cls_color = filtered_metadata_views["class_color"].tolist()
            cells_subclass = filtered_metadata_views["subclass"].tolist()
            cells_subcls_color = filtered_metadata_views["subclass_color"].tolist()
            cells_supertype = filtered_metadata_views["supertype"].tolist()
            cells_supertype_color = filtered_metadata_views["supertype_color"].tolist()
            cells_cluster = filtered_metadata_views["cluster"].tolist()
            cells_cluster_color = filtered_metadata_views["cluster_color"].tolist()

            unique_cells_class, unique_indices = np.unique(cells_class, return_index=True)
            unique_cells_cls_color = np.array(cells_cls_color)[unique_indices]

            unique_cells_subclass, unique_indices = np.unique(cells_subclass, return_index=True)
            unique_cells_subcls_color = np.array(cells_subcls_color)[unique_indices]

            unique_cells_supertype, unique_indices = np.unique(cells_supertype, return_index=True)
            unique_cells_supertype_color = np.array(cells_supertype_color)[unique_indices]

            unique_cells_cluster, unique_indices = np.unique(cells_cluster, return_index=True)
            unique_cells_cluster_color = np.array(cells_cluster_color)[unique_indices]

            non_neuronal_cell_types = ["Astro", "Oligo", "Vascular", "Immune", "Epen"]
            neuronal_mask = np.array(
                [True if any([j in i for j in non_neuronal_cell_types]) else False for i in cells_class])
            neuronal_mask_2 = np.array(
                [True if any([j in i for j in non_neuronal_cell_types]) else False for i in unique_cells_class])

            ########################################################################################################################
            # Color transformed points
            ########################################################################################################################

            # gubra ref: 369, 512, 268
            reference_file = r"C:\Users\MANMONTALCINI\PycharmProjects\ClearMap2\ClearMap\Resources\Atlas\gubra_template.tif"
            reference = tifffile.imread(reference_file)

            for cc, ccat, cn in zip([cells_cls_color, cells_subcls_color, cells_supertype_color, cells_cluster_color],
                                    [cells_class, cells_subclass, cells_supertype, cells_cluster],
                                    ["class", "subclass", "supertype", "cluster"]):
                # Horizontal
                fig = plt.figure()
                ax = plt.subplot(111)
                if filtered_points.size > 0:
                    ax.scatter(filtered_points[:, 2][~neuronal_mask], filtered_points[:, 0][~neuronal_mask],
                               c=np.array(cc)[~neuronal_mask], s=1,
                               lw=0., edgecolors="black", alpha=1)
                ax.imshow(np.rot90(np.max(reference, axis=2))[::-1], cmap='gray_r', alpha=0.3)
                ax.set_xlim(0, 369)
                ax.set_ylim(0, 512)
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                plt.savefig(os.path.join(whole_brain_dir, f"neurons_horizontal_{cn}_{p}.png"), dpi=300)
                # plt.show()
                plt.close()

                fig = plt.figure()
                ax = plt.subplot(111)
                if filtered_points.size > 0:
                    ax.scatter(filtered_points[:, 2], filtered_points[:, 0], c=np.array(cc), s=1,
                               lw=0., edgecolors="black", alpha=1)
                ax.imshow(np.rot90(np.max(reference, axis=2))[::-1], cmap='gray_r', alpha=0.3)
                ax.set_xlim(0, 369)
                ax.set_ylim(0, 512)
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                plt.savefig(os.path.join(whole_brain_dir, f"all_horizontal_{cn}_{p}.png"), dpi=300)
                # plt.show()
                plt.close()

                # Sagittal
                fig = plt.figure()
                ax = plt.subplot(111)
                if filtered_points.size > 0:
                    ax.scatter(filtered_points[:, 0][~neuronal_mask], filtered_points[:, 1][~neuronal_mask],
                               c=np.array(cc)[~neuronal_mask], s=1,
                               lw=0., edgecolors="black", alpha=1)
                ax.imshow(np.rot90(np.max(reference, axis=0))[::-1], cmap='gray_r', alpha=0.3)
                ax.set_xlim(0, 512)
                ax.set_ylim(0, 268)
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                plt.savefig(os.path.join(whole_brain_dir, f"neurons_sagittal_{cn}_{p}.png"), dpi=300)
                # plt.show()
                plt.close()

                fig = plt.figure()
                ax = plt.subplot(111)
                if filtered_points.size > 0:
                    ax.scatter(filtered_points[:, 0], filtered_points[:, 1], c=np.array(cc), s=1,
                               lw=0., edgecolors="black", alpha=1)
                ax.imshow(np.rot90(np.max(reference, axis=0))[::-1], cmap='gray_r', alpha=0.3)
                ax.set_xlim(0, 512)
                ax.set_ylim(0, 268)
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                plt.savefig(os.path.join(whole_brain_dir, f"all_sagittal_{cn}_{p}.png"), dpi=300)
                # plt.show()
                plt.close()

                # Coronal
                fig = plt.figure()
                ax = plt.subplot(111)
                if filtered_points.size > 0:
                    ax.scatter(filtered_points[:, 2][~neuronal_mask], filtered_points[:, 1][~neuronal_mask],
                               c=np.array(cc)[~neuronal_mask], s=1,
                               lw=0., edgecolors="black", alpha=1)
                ax.imshow(np.rot90(np.max(reference, axis=1))[::-1], cmap='gray_r', alpha=0.3)
                ax.set_xlim(0, 369)
                ax.set_ylim(0, 268)
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                plt.savefig(os.path.join(whole_brain_dir, f"neurons_coronal_{cn}_{p}.png"), dpi=300)
                # plt.show()
                plt.close()

                ############################################################################################################
                # Focus on PB
                ############################################################################################################

                # PB (374 - 417)
                saving_slices_dir = os.path.join(res_dir, "slices")
                if not os.path.exists(saving_slices_dir):
                    os.mkdir(saving_slices_dir)
                saving_slices_pdf_dir = os.path.join(res_dir, "slices_pdf")
                if not os.path.exists(saving_slices_pdf_dir):
                    os.mkdir(saving_slices_pdf_dir)

                with PdfPages(os.path.join(saving_slices_pdf_dir, f"{cn}_{p}.pdf")) as pdf:

                    fig = plt.figure()
                    ax = plt.subplot(111)
                    if filtered_points.size > 0:
                        if s == 0:
                            s_n = s
                        else:
                            s_n = s - thickness_dif

                        if e == end:
                            e_n = e
                        else:
                            e_n = e + thickness_dif
                        filtered_points_plane, mask_plane = filter_coordinates_dim(filtered_points[~neuronal_mask],
                                                                                   s_n, e_n, 0)
                        categories = np.array(ccat)[~neuronal_mask][mask_plane]
                    else:
                        filtered_points_plane, mask_plane = np.array([]), np.array([])
                        categories = np.array([])
                    if cn != "class":

                        # PB_categories_mask = np.array([True if i.split(" ")[1].startswith("PB") else False for i in PB_categories])
                        categories_mask = np.ones_like(categories, dtype=bool)
                        if categories_mask.size > 0:
                            categories = categories[categories_mask]
                            print(categories)
                            if filtered_points_plane[categories_mask].size > 0:
                                ax.scatter(filtered_points_plane[:, 2][categories_mask],
                                           filtered_points_plane[:, 1][categories_mask],
                                           c=np.array(cc)[~neuronal_mask][mask_plane][categories_mask], s=1,
                                           lw=0., edgecolors="black", alpha=1)
                    else:
                        if filtered_points_plane.size > 0:
                            ax.scatter(filtered_points_plane[:, 2],
                                       filtered_points_plane[:, 1],
                                       c=np.array(cc)[~neuronal_mask][mask_plane], s=1,
                                       lw=0., edgecolors="black", alpha=1)
                    PB_plane = (s + e) / 2
                    ax.imshow(np.rot90(reference[:, int(PB_plane), :])[::-1], cmap='gray_r', alpha=0.3)
                    ax.set_xlim(0, 369)
                    ax.set_ylim(0, 268)
                    ax.invert_yaxis()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plt.savefig(os.path.join(res_dir, f"neurons_coronal_PB_{cn}_{v}.png"), dpi=300)
                    plt.savefig(os.path.join(saving_slices_dir, f"{cn}_{p}.png"), dpi=300)
                    pdf.savefig()
                    plt.close()

                    # Save legends
                    fig = plt.figure()
                    ax = plt.subplot(111)
                    unique_cat, idx_cat = np.unique(categories, return_index=True)
                    if np.array(cc).size > 0:
                        if cn != "class":
                            if categories_mask.size > 0:
                                handles = [
                                    mlines.Line2D([0], [0], color=c, marker='o', linestyle='None', markersize=5)
                                    for c in np.array(cc)[~neuronal_mask][mask_plane][categories_mask][idx_cat]]
                        else:
                            handles = [mlines.Line2D([0], [0], color=c, marker='o', linestyle='None', markersize=5)
                                       for c in np.array(cc)[~neuronal_mask][mask_plane][idx_cat]]
                    else:
                        handles = []
                    ncol = int(len(handles)/15) + 1
                    if len(unique_cat) > 0:
                        ax.legend(handles=handles, labels=list(unique_cat), fontsize=5, ncol=ncol, loc=2)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    pdf.savefig()
                    plt.close()

                    # Bar plot counts
                    if np.array(ccat).size > 0:
                        color_counts = pd.Series(categories).value_counts() \
                            .reindex(np.array(ccat)[~neuronal_mask][mask_plane][idx_cat], fill_value=0)
                        # Create a DataFrame for sorting
                        if cn != "class":
                            count_df = pd.DataFrame({
                                'Color': np.array(cc)[~neuronal_mask][mask_plane][categories_mask][idx_cat],
                                'Label': unique_cat,
                                'Count': color_counts
                            })
                        else:
                            count_df = pd.DataFrame({
                                'Color': np.array(cc)[~neuronal_mask][mask_plane][idx_cat],
                                'Label': unique_cat,
                                'Count': color_counts
                            })
                        # Sort the DataFrame by counts in descending order
                        sorted_df = count_df.sort_values(by='Count', ascending=False)

                        fig = plt.figure()
                        ax = plt.subplot(111)
                        ax.bar(range(len(unique_cat)), sorted_df['Count'], color=sorted_df['Color'])
                        ax.set_ylabel('Number of Cells', fontsize=6)
                        ax.set_xticks(range(len(sorted_df)))
                        ax.set_xticklabels(sorted_df['Label'], rotation=45, fontsize=6)
                        plt.yticks(fontsize=6)
                        pdf.savefig()
                        plt.close()

                    else:
                        fig = plt.figure()
                        ax = plt.subplot(111)
                        ax.set_ylabel('Number of Cells', fontsize=6)
                        plt.yticks(fontsize=6)
                        pdf.savefig()
                        plt.close()
