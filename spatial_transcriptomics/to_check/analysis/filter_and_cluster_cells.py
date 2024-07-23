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


# matplotlib.use("Qt5Agg")

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

dataset_n = 1
dataset_id = f"Zhuang-ABCA-{dataset_n}"
download_base = r'/mnt/data/spatial_transcriptomics'
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
transformed_coordinates = np.load(r"/mnt/data/spatial_transcriptomics/results/transformed_cells_to_gubra/"
                                  fr"general/all_transformed_cells_{dataset_n}.npy")

# maps_path = r"/mnt/data/spatial_transcriptomics/results/whole_brain"
# maps = ["whole_brain"]
maps_path = r"/mnt/data/spatial_transcriptomics/results/mpd5_pick1"
maps = ["Pick-1_vs_Vehicle"]

for m in maps:

    map_path = os.path.join(maps_path, m)
    if os.path.isdir(map_path):
        res_dir = os.path.join(map_path, "transcriptomics")
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
        # mask = tifffile.imread(os.path.join(map_path, r"bin.tif"))
        mask = tifffile.imread(os.path.join(map_path, r"bin_whole.tif"))
        if mask.dtype == "bool":
            mask = mask.astype("uint8")*255

        print("Filtering points in mask!")
        filtered_points, mask_point = filter_points_in_3d_mask(transformed_coordinates, mask)
        filtered_labels = cell_labels[::-1][mask_point]

        adata_filtered = adata[adata.obs.index.isin(filtered_labels), :]
        obs_df = adata_filtered.obs.copy()
        # Reorder obs_df according to filtered_labels
        obs_df = obs_df.loc[filtered_labels]
        # Use the ordered index from obs_df to index into adata
        adata_filtered = adata[obs_df.index, :]
        s = time.time()
        gdata = adata_filtered.to_df()
        e = time.time()
        print(f"Function run time: {e - s} seconds")  # Print the elapsed time
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
        cells_neurotransmitter = filtered_metadata_views["neurotransmitter"].tolist()
        cells_neurotransmitter_color = filtered_metadata_views["neurotransmitter_color"].tolist()
        cells_class = filtered_metadata_views["class"].tolist()
        cells_cls_color = filtered_metadata_views["class_color"].tolist()
        cells_subclass = filtered_metadata_views["subclass"].tolist()
        cells_subcls_color = filtered_metadata_views["subclass_color"].tolist()
        cells_supertype = filtered_metadata_views["supertype"].tolist()
        cells_supertype_color = filtered_metadata_views["supertype_color"].tolist()
        cells_cluster = filtered_metadata_views["cluster"].tolist()
        cells_cluster_color = filtered_metadata_views["cluster_color"].tolist()

        unique_cells_nt, unique_indices = np.unique(cells_neurotransmitter, return_index=True)
        unique_cells_nt_color = np.array(cells_neurotransmitter_color)[unique_indices]

        unique_cells_class, unique_indices = np.unique(cells_class, return_index=True)
        unique_cells_cls_color = np.array(cells_cls_color)[unique_indices]

        unique_cells_subclass, unique_indices = np.unique(cells_subclass, return_index=True)
        unique_cells_subcls_color = np.array(cells_subcls_color)[unique_indices]

        unique_cells_supertype, unique_indices = np.unique(cells_supertype, return_index=True)
        unique_cells_supertype_color = np.array(cells_supertype_color)[unique_indices]

        unique_cells_cluster, unique_indices = np.unique(cells_cluster, return_index=True)
        unique_cells_cluster_color = np.array(cells_cluster_color)[unique_indices]

        # umap_embeddings, hdbscan_clusters = umap_hdbscan_clustering(gdata, return_hdbscan=False)

        non_neuronal_cell_types = ["Astro", "Oligo", "Vascular", "Immune", "Epen"]
        neuronal_mask = np.array(
            [True if any([j in i for j in non_neuronal_cell_types]) else False for i in cells_class])
        neuronal_mask_2 = np.array(
            [True if any([j in i for j in non_neuronal_cell_types]) else False for i in unique_cells_class])

        # plot_umap_with_clusters(umap_embeddings[~neuronal_mask],
        #                         np.array(cells_cls_color)[~neuronal_mask],
        #                         title=f"Class",
        #                         legend=[unique_cells_cls_color[~neuronal_mask_2],
        #                                 unique_cells_class[~neuronal_mask_2]],
        #                         alpha=0.5,
        #                         saving_dir=os.path.join(res_dir, "umap_neurons.png"))
        #
        # ########################################################################################################################
        # # Custom umap
        # ########################################################################################################################
        #
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [3, 1]})
        # # UMAP plot on the first subplot
        # scatter = ax1.scatter(umap_embeddings["UMAP1"][~neuronal_mask], umap_embeddings["UMAP2"][~neuronal_mask],
        #                       c=np.array(cells_cls_color)[~neuronal_mask], s=3, alpha=1)
        # scatter = ax1.scatter(umap_embeddings["UMAP1"][neuronal_mask], umap_embeddings["UMAP2"][neuronal_mask],
        #                       c=np.array(cells_cls_color)[neuronal_mask], s=3, alpha=0.01)
        # ax1.set_aspect('equal', 'datalim')
        # ax1.set_title('UMAP projection', fontsize=12)
        # ax1.set_xlabel('UMAP1')
        # ax1.set_ylabel('UMAP2')
        # # Count the occurrences of each color in labels
        # color_counts = pd.Series(np.array(cells_cls_color)[~neuronal_mask]).value_counts() \
        #     .reindex(unique_cells_cls_color[~neuronal_mask_2], fill_value=0)
        # # Create a DataFrame for sorting
        # count_df = pd.DataFrame({
        #     'Color': unique_cells_cls_color[~neuronal_mask_2],
        #     'Label': unique_cells_class[~neuronal_mask_2],
        #     'Count': color_counts
        # })
        # # Sort the DataFrame by counts in descending order and select top 20
        # sorted_df = count_df.sort_values(by='Count', ascending=False).head(20)
        # # Update the legend to reflect top 20 items
        # top_20_handles = [mlines.Line2D([0], [0], color=c, marker='o', linestyle='None', markersize=10)
        #                   for c in sorted_df['Color']]
        # top_20_labels = sorted_df['Label'].tolist()
        # ax1.legend(handles=top_20_handles, labels=top_20_labels, fontsize=8, ncol=1, loc=2)
        #
        # # Bar plot on the second subplot for top 20 items
        # ax2.bar(range(len(sorted_df)), sorted_df['Count'], color=sorted_df['Color'])
        # ax2.set_ylabel('Number of Cells')
        # ax2.set_xticks(range(len(sorted_df)))
        # ax2.set_xticklabels(sorted_df['Label'], rotation=90, fontsize=8)
        #
        # # Adjust the layout
        # plt.tight_layout()
        # plt.savefig(os.path.join(res_dir, "umap_neurons.png"), dpi=300)
        # # plt.show()
        #
        # ########################################################################################################################
        # # Bar plot PB
        # ########################################################################################################################
        #
        # plt.rcParams['ytick.labelsize'] = 20
        # plt.rcParams['xtick.labelsize'] = 20
        #
        #
        # def bar_plot(cells_color, unique_cells, unique_cells_color, saving_dir):
        #     fig, (ax1) = plt.subplots(1, 1, figsize=(15, 10))
        #
        #     # Count the occurrences of each color in labels
        #     color_counts = pd.Series(np.array(cells_color)).value_counts() \
        #         .reindex(unique_cells_color, fill_value=0)
        #     # Create a DataFrame for sorting
        #     count_df = pd.DataFrame({
        #         'Color': unique_cells_color,
        #         'Label': unique_cells,
        #         'Count': color_counts
        #     })
        #
        #     # Sort the DataFrame by counts in descending order and select top 10
        #     sorted_df = count_df.sort_values(by='Count', ascending=False).head(10)
        #
        #     # Bar plot on the second subplot for top 20 items
        #     ax1.bar(range(len(sorted_df)), sorted_df['Count'], color=sorted_df['Color'])
        #     ax1.set_ylabel('Number of Cells', fontsize=20)
        #     ax1.set_xlim(-0.5, 10)
        #     # ax1.set_xticks(range(len(sorted_df)))
        #     ax1.set_xticks([])
        #     # ax1.set_xticklabels(sorted_df['Label'], rotation=45, fontsize=15)
        #
        #     top_10_handles = [mlines.Line2D([0], [0], color=c, marker='o', linestyle='None', markersize=10)
        #                       for c in sorted_df['Color']]
        #     top_10_labels = sorted_df['Label'].tolist()
        #     ax1.legend(handles=top_10_handles, labels=top_10_labels, fontsize=20, ncol=1, loc=1)
        #
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(saving_dir, "bar_plot_all.png"), dpi=300)
        #     # plt.show()
        #
        #
        # ########################################################################################################################
        # # Standard umap
        # ########################################################################################################################
        #
        # plot_umap_with_clusters(umap_embeddings,
        #                         np.array(cells_cls_color),
        #                         title=f"Class",
        #                         legend=[unique_cells_cls_color,
        #                                 unique_cells_class],
        #                         alpha=1,
        #                         saving_dir=os.path.join(res_dir, "umap_class.png"))
        # plot_umap_with_clusters(umap_embeddings, np.array(cells_subcls_color), title=f"Subclass",
        #                         legend=[unique_cells_subcls_color, unique_cells_subclass],
        #                         saving_dir=os.path.join(res_dir, "umap_subclass.png"))
        # plot_umap_with_clusters(umap_embeddings, np.array(cells_supertype_color), title=f"Supertype",
        #                         legend=[unique_cells_supertype_color, unique_cells_supertype],
        #                         saving_dir=os.path.join(res_dir, "umap_supertype.png"))
        # plot_umap_with_clusters(umap_embeddings, np.array(cells_cluster_color), title=f"Cluster",
        #                         legend=[unique_cells_cluster_color, unique_cells_cluster],
        #                         saving_dir=os.path.join(res_dir, "umap_cluster.png"))

        ########################################################################################################################
        # Color transformed points
        ########################################################################################################################

        # gubra ref: 369, 512, 268
        reference_file = r"/mnt/data/spatial_transcriptomics/atlas_ressources/gubra_template_sagittal.tif"
        reference = tifffile.imread(reference_file)
        cell_color = "blue"

        for cc, ccat, cn in zip([cells_neurotransmitter_color, cells_cls_color, cells_subcls_color,
                                 cells_supertype_color, cells_cluster_color],
                                [cells_neurotransmitter, cells_class, cells_subclass,
                                 cells_supertype, cells_cluster],
                                ["neurotransmitter", "class", "subclass", "supertype", "cluster"]):
            # Horizontal
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.scatter(filtered_points[:, 2][~neuronal_mask], filtered_points[:, 0][~neuronal_mask],
                       c=np.array(cc)[~neuronal_mask],
                       # c=cell_color,
                       s=1,
                       lw=0., edgecolors="black", alpha=1)
            ax.imshow(np.rot90(np.max(reference, axis=2))[::-1], cmap='gray_r', alpha=0.3)
            ax.set_xlim(0, 369)
            ax.set_ylim(0, 512)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(os.path.join(res_dir, f"neurons_horizontal_{cn}.png"), dpi=300)
            # plt.show()

            fig = plt.figure()
            ax = plt.subplot(111)
            ax.scatter(filtered_points[:, 2], filtered_points[:, 0],
                       c=np.array(cc),
                       # c=cell_color,
                       s=1,
                       lw=0., edgecolors="black", alpha=1)
            ax.imshow(np.rot90(np.max(reference, axis=2))[::-1], cmap='gray_r', alpha=0.3)
            ax.set_xlim(0, 369)
            ax.set_ylim(0, 512)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(os.path.join(res_dir, f"all_horizontal_{cn}.png"), dpi=300)
            # plt.show()

            # Sagittal
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.scatter(filtered_points[:, 0][~neuronal_mask], filtered_points[:, 1][~neuronal_mask],
                       c=np.array(cc)[~neuronal_mask],
                       # c=cell_color,
                       s=1,
                       lw=0., edgecolors="black", alpha=1)
            ax.imshow(np.rot90(np.max(reference, axis=0))[::-1], cmap='gray_r', alpha=0.3)
            ax.set_xlim(0, 512)
            ax.set_ylim(0, 268)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(os.path.join(res_dir, f"neurons_sagittal_{cn}.png"), dpi=300)
            # plt.show()

            fig = plt.figure()
            ax = plt.subplot(111)
            ax.scatter(filtered_points[:, 0], filtered_points[:, 1],
                       c=np.array(cc),
                       # c=cell_color,
                       s=1,
                       lw=0., edgecolors="black", alpha=1)
            ax.imshow(np.rot90(np.max(reference, axis=0))[::-1], cmap='gray_r', alpha=0.3)
            ax.set_xlim(0, 512)
            ax.set_ylim(0, 268)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(os.path.join(res_dir, f"all_sagittal_{cn}.png"), dpi=300)
            # plt.show()

            # Coronal
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.scatter(filtered_points[:, 2][~neuronal_mask], filtered_points[:, 1][~neuronal_mask],
                       c=np.array(cc)[~neuronal_mask],
                       # c=cell_color,
                       s=1,
                       lw=0., edgecolors="black", alpha=1)
            ax.imshow(np.rot90(np.max(reference, axis=1))[::-1], cmap='gray_r', alpha=0.3)
            ax.set_xlim(0, 369)
            ax.set_ylim(0, 268)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(os.path.join(res_dir, f"neurons_coronal_{cn}.png"), dpi=300)
            # plt.show()

            fig = plt.figure()
            ax = plt.subplot(111)
            ax.scatter(filtered_points[:, 2], filtered_points[:, 1],
                       c=np.array(cc),
                       # c=cell_color,
                       s=1,
                       lw=0., edgecolors="black", alpha=1)
            ax.imshow(np.rot90(np.max(reference, axis=1))[::-1], cmap='gray_r', alpha=0.3)
            ax.set_xlim(0, 369)
            ax.set_ylim(0, 268)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(os.path.join(res_dir, f"all_coronal_{cn}.png"), dpi=300)
            # plt.show()

            ############################################################################################################
            # Focus on PB
            ############################################################################################################

            # PB (374 - 417)
            saving_slices_dir = os.path.join(res_dir, "slices")
            if not os.path.exists(saving_slices_dir):
                os.mkdir(saving_slices_dir)
            # start_v, stop_v = 374, 417
            start_v, stop_v = 0, 512
            range_v = np.arange(start_v, stop_v, 5)

            for v in range(len(range_v) - 1):

                with PdfPages(os.path.join(saving_slices_dir, f"{cn}_{v}.pdf")) as pdf:

                    fig = plt.figure()
                    ax = plt.subplot(111)
                    filtered_points_plane, mask_plane = filter_coordinates_dim(filtered_points[~neuronal_mask],
                                                                               range_v[v], range_v[v + 1], 0)
                    categories = np.array(ccat)[~neuronal_mask][mask_plane]
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
                    PB_plane = (range_v[v] + range_v[v + 1]) / 2
                    ax.imshow(np.rot90(reference[:, int(PB_plane), :])[::-1], cmap='gray_r', alpha=0.3)
                    ax.set_xlim(0, 369)
                    ax.set_ylim(0, 268)
                    ax.invert_yaxis()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plt.savefig(os.path.join(res_dir, f"neurons_coronal_PB_{cn}_{v}.png"), dpi=300)
                    plt.savefig(os.path.join(saving_slices_dir, f"{cn}_{v}.png"), dpi=300)
                    pdf.savefig()
                    plt.close()

                    # Save legends
                    fig = plt.figure()
                    ax = plt.subplot(111)
                    unique_cat, idx_cat = np.unique(categories, return_index=True)
                    if cn != "class":
                        if categories_mask.size > 0:
                            handles = [
                                mlines.Line2D([0], [0], color=c, marker='o', linestyle='None', markersize=5)
                                for c in np.array(cc)[~neuronal_mask][mask_plane][categories_mask][idx_cat]]
                    else:
                        handles = [mlines.Line2D([0], [0], color=c, marker='o', linestyle='None', markersize=5)
                                   for c in np.array(cc)[~neuronal_mask][mask_plane][idx_cat]]
                    ncol = int(len(handles)/15) + 1
                    ax.legend(handles=handles, labels=list(unique_cat), fontsize=5, ncol=ncol, loc=2)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    pdf.savefig()
                    plt.close()

                    # Bar plot counts
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

            ############################################################################################################
            # END
            ############################################################################################################

            fig = plt.figure()
            ax = plt.subplot(111)
            ax.scatter(filtered_points[:, 2], filtered_points[:, 1], c=np.array(cc), s=1,
                       lw=0.1, edgecolors="black", alpha=1)
            ax.imshow(np.rot90(np.max(reference, axis=1))[::-1], cmap='gray_r', alpha=0.3)
            ax.set_xlim(0, 369)
            ax.set_ylim(0, 268)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(os.path.join(res_dir, f"all_coronal_{cn}.png"), dpi=300)
            # plt.show()

        # gubra ref: 369, 512, 268
        # reference_file = r"/mnt/data/spatial_transcriptomics/atlas_ressources/gubra_template_horizontal.tif"
        reference = tifffile.imread(reference_file)

        ########################################################################################################################
        # Volumes for all classes
        ########################################################################################################################

        # break

        volume = reference.copy() / 5
        volume = volume.astype("uint8")
        volume = 255 - volume
        volume = np.swapaxes(volume, 0, 1)
        volume = np.swapaxes(volume, 1, 2)
        volume_rgb = np.stack([volume, volume, volume], axis=-1)
        # volume_rgba[:, :, :, 3] = 100
        all_classes_dummy = volume_rgb.copy()

        # Class
        if unique_cells_class.size != 0:
            n_classes = len(unique_cells_class[~neuronal_mask_2])
            print(f"[{m}] {n_classes} unique cell classes found")
            class_folder = check_and_create_folder(os.path.join(res_dir, "class"))
            for i, (cls_name, cls_color) in enumerate(zip(unique_cells_class[~neuronal_mask_2],
                                                          unique_cells_cls_color[~neuronal_mask_2])):
                class_mask = np.logical_and(np.array(cells_class) == cls_name, ~neuronal_mask)
                if not np.sum(class_mask) == 0:
                    unique_class_folder = check_and_create_folder(os.path.join(class_folder, cls_name))
                    print("\n")
                    print(f"[{m}] Creating volume for class: {cls_name}; {i}/{n_classes}")
                    print(f"[{m}] {np.sum(class_mask)} cells detected")
                    class_points = filtered_points[class_mask]

                    # Subclass
                    subcls = np.array(cells_subclass)[class_mask]
                    subcls_c = np.array(cells_subcls_color)[class_mask]
                    subcls_u, idx = np.unique(subcls, return_index=True)
                    subcls_c_u = subcls_c[idx]
                    n_subclasses = len(subcls_u)
                    for j, (subcls_name, subcls_color) in enumerate(zip(subcls_u, subcls_c_u)):
                        subcls_name = subcls_name.replace("/", "")
                        # subclass_mask = np.array(cells_subclass) == subcls_name
                        subclass_mask = np.logical_and(np.array(cells_subclass) == subcls_name, ~neuronal_mask)
                        if not np.sum(subclass_mask) == 0:
                            unique_subclass_folder = check_and_create_folder(
                                os.path.join(unique_class_folder, subcls_name))
                            print(f"[{m}] Creating volume for subclass: {subcls_name}; {j}/{n_subclasses}")
                            print(f"[{m}] {np.sum(subclass_mask)} cells detected")
                            subclass_points = filtered_points[subclass_mask]

                            # Supertype
                            supertp = np.array(cells_supertype)[subclass_mask]
                            supertp_c = np.array(cells_supertype_color)[subclass_mask]
                            supertp_u, idx = np.unique(supertp, return_index=True)
                            supertp_c_u = supertp_c[idx]
                            n_supertypes = len(supertp_u)
                            for k, (supertp_name, supertp_color) in enumerate(zip(supertp_u, supertp_c_u)):
                                supertp_name = supertp_name.replace("/", "")
                                # supertp_mask = np.array(cells_supertype) == supertp_name
                                supertp_mask = np.logical_and(np.array(cells_supertype) == supertp_name, ~neuronal_mask)
                                if not np.sum(supertp_mask) == 0:
                                    unique_supertype_folder = check_and_create_folder(
                                        os.path.join(unique_subclass_folder, supertp_name))
                                    # print(f"[{m}] Creating volume for supertype: {supertp_name}; {k}/{n_supertypes}")
                                    # print(f"[{m}] {np.sum(supertp_mask)} cells detected")
                                    supertp_points = filtered_points[supertp_mask]

                                    # Cluster
                                    clst = np.array(cells_cluster)[supertp_mask]
                                    clst_c = np.array(cells_cluster_color)[supertp_mask]
                                    clst_u, idx = np.unique(clst, return_index=True)
                                    clst_c_u = clst_c[idx]
                                    n_clusters = len(clst_u)
                                    for l, (clst_name, clst_color) in enumerate(zip(clst_u, clst_c_u)):
                                        clst_name = clst_name.replace("/", "")
                                        # clst_mask = np.array(cells_cluster) == clst_name
                                        clst_mask = np.logical_and(np.array(cells_cluster) == clst_name, ~neuronal_mask)
                                        if not np.sum(clst_mask) == 0:
                                            unique_cluster_folder = check_and_create_folder(
                                                os.path.join(unique_supertype_folder, clst_name))
                                            # print(f"[{m}] Creating volume for cluster: {clst_name}; {l}/{n_clusters}")
                                            # print(f"[{m}] {np.sum(clst_mask)} cells detected")
                                            clst_points = filtered_points[clst_mask]

                                            # dummy = volume_rgb.copy()
                                            # # dummy[tuple(clst_points.astype(int).T)] = (0, 0, 0)
                                            # dummy[tuple(clst_points.astype(int).T)] = hex_to_rgb(clst_color)
                                            # tifffile.imwrite(os.path.join(unique_cluster_folder, "volume.tif"), dummy)
                                            # max_proj = np.min(dummy, axis=1)
                                            # tifffile.imwrite(os.path.join(unique_cluster_folder, "max_proj.tif"), max_proj,
                                            #                  compress=0)

                                            fig = plt.figure()
                                            ax = plt.subplot(111)
                                            ax.imshow(np.min(volume_rgb, axis=1))
                                            ax.scatter(clst_points[:, 2], clst_points[:, 0], c=clst_color, s=1,
                                                       lw=0.1, edgecolors="black", alpha=1)
                                            remove_spines_and_ticks(ax)
                                            plt.tight_layout()
                                            plt.savefig(os.path.join(unique_cluster_folder, "max_proj.png"), dpi=500)

                                            # Save the data
                                            df_points = pd.DataFrame(clst_points, columns=['x', 'y', 'z'])
                                            df_points['color'] = clst_color
                                            df_points.to_csv(os.path.join(unique_cluster_folder, 'points.csv'),
                                                             index=False)

                                            # # Coronal PB
                                            # max_proj_PB = np.min(dummy[372:416], axis=0)
                                            # PB_plane = dummy[394]
                                            # # PB_plane[np.all(max_proj_PB == 0, axis=2)] = (0, 0, 0)
                                            # PB_plane[np.all(max_proj_PB == 0, axis=2)] = hex_to_rgb(clst_color)
                                            # tifffile.imwrite(os.path.join(unique_cluster_folder, "cor_proj_PB.tif"), PB_plane[136:179, 110:166],
                                            #                  compress=0)

                                    # dummy = volume_rgb.copy()
                                    # # dummy[tuple(supertp_points.astype(int).T)] = (0, 0, 0)
                                    # dummy[tuple(supertp_points.astype(int).T)] = hex_to_rgb(supertp_color)
                                    # tifffile.imwrite(os.path.join(unique_supertype_folder, "volume.tif"), dummy)
                                    # max_proj = np.min(dummy, axis=1)
                                    # tifffile.imwrite(os.path.join(unique_supertype_folder, "max_proj.tif"), max_proj,
                                    #                  compress=0)

                                    fig = plt.figure()
                                    ax = plt.subplot(111)
                                    ax.imshow(np.min(volume_rgb, axis=1))
                                    ax.scatter(supertp_points[:, 2], supertp_points[:, 0], c=supertp_color, s=1,
                                               lw=0.1, edgecolors="black", alpha=1)
                                    remove_spines_and_ticks(ax)
                                    plt.tight_layout()
                                    plt.savefig(os.path.join(unique_supertype_folder, "max_proj.png"), dpi=500)

                                    # Save the data
                                    df_points = pd.DataFrame(supertp_points, columns=['x', 'y', 'z'])
                                    df_points['color'] = supertp_color
                                    df_points.to_csv(os.path.join(unique_supertype_folder, 'points.csv'), index=False)

                                    # # Coronal PB
                                    # max_proj_PB = np.min(dummy[372:416], axis=0)
                                    # PB_plane = dummy[394]
                                    # # PB_plane[np.all(max_proj_PB == 0, axis=2)] = (0, 0, 0)
                                    # PB_plane[np.all(max_proj_PB == 0, axis=2)] = hex_to_rgb(supertp_color)
                                    # tifffile.imwrite(os.path.join(unique_supertype_folder, "cor_proj_PB.tif"), PB_plane[136:179, 110:166],
                                    #                  compress=0)
                                    bar_plot(clst_c, clst_u, clst_c_u, unique_supertype_folder)

                            # dummy = volume_rgb.copy()
                            # dummy[tuple(subclass_points.astype(int).T)] = hex_to_rgb(subcls_color)
                            # dummy[tuple(subclass_points.astype(int).T)] = (0, 0, 0)
                            # tifffile.imwrite(os.path.join(unique_subclass_folder, "volume.tif"), dummy)
                            # max_proj = np.min(dummy, axis=1)
                            # tifffile.imwrite(os.path.join(unique_subclass_folder, "max_proj.tif"), max_proj,
                            #                  compress=0)

                            fig = plt.figure()
                            ax = plt.subplot(111)
                            ax.imshow(np.min(volume_rgb, axis=1))
                            ax.scatter(subclass_points[:, 2], subclass_points[:, 0], c=subcls_color, s=1,
                                       lw=0.1, edgecolors="black", alpha=1)
                            remove_spines_and_ticks(ax)
                            plt.tight_layout()
                            plt.savefig(os.path.join(unique_subclass_folder, "max_proj.png"), dpi=500)

                            # Save the data
                            df_points = pd.DataFrame(subclass_points, columns=['x', 'y', 'z'])
                            df_points['color'] = subcls_color
                            df_points.to_csv(os.path.join(unique_subclass_folder, 'points.csv'), index=False)

                            # # Coronal PB
                            # max_proj_PB = np.min(dummy[372:416], axis=0)
                            # PB_plane = dummy[394]
                            # # PB_plane[np.all(max_proj_PB == 0, axis=2)] = (0, 0, 0)
                            # PB_plane[np.all(max_proj_PB == 0, axis=2)] = hex_to_rgb(subcls_color)
                            # tifffile.imwrite(os.path.join(unique_subclass_folder, "cor_proj_PB.tif"), PB_plane[136:179, 110:166],
                            #                  compress=0)
                            bar_plot(supertp_c, supertp_u, supertp_c_u, unique_subclass_folder)

                    # dummy = volume_rgb.copy()
                    # # dummy[tuple(class_points.astype(int).T)] = (0, 0, 0)
                    # dummy[tuple(class_points.astype(int).T)] = hex_to_rgb(cls_color)
                    # all_classes_dummy[tuple(class_points.astype(int).T)] = hex_to_rgb(cls_color)
                    # # dummy[tuple(class_points.astype(int).T)] = (0, 0, 0)
                    # tifffile.imwrite(os.path.join(unique_class_folder, "volume.tif"), dummy)
                    # max_proj = np.min(dummy, axis=1)

                    fig = plt.figure()
                    ax = plt.subplot(111)
                    ax.imshow(np.min(volume_rgb, axis=1))
                    ax.scatter(class_points[:, 2], class_points[:, 0], c=cls_color, s=1,
                               lw=0.1, edgecolors="black", alpha=1)
                    remove_spines_and_ticks(ax)
                    plt.tight_layout()
                    plt.savefig(os.path.join(unique_class_folder, "max_proj.png"), dpi=500)

                    # Save the data
                    df_points = pd.DataFrame(class_points, columns=['x', 'y', 'z'])
                    df_points['color'] = cls_color
                    df_points.to_csv(os.path.join(unique_class_folder, 'points.csv'), index=False)

                    # tifffile.imwrite(os.path.join(unique_class_folder, "max_proj.tif"), max_proj, compress=0)

                    # # Coronal PB
                    # max_proj_PB = np.min(dummy[372:416], axis=0)
                    # PB_plane = dummy[394]
                    # # PB_plane[np.all(max_proj_PB == 0, axis=2)] = (0, 0, 0)
                    # PB_plane[np.all(max_proj_PB == 0, axis=2)] = hex_to_rgb(cls_color)
                    # tifffile.imwrite(os.path.join(unique_class_folder, "cor_proj_PB.tif"), PB_plane[136:179, 110:166])
                    bar_plot(subcls_c, subcls_u, subcls_c_u, unique_class_folder)
            # tifffile.imwrite(os.path.join(class_folder, "volume.tif"), all_classes_dummy)
            bar_plot(np.array(cells_cls_color)[~neuronal_mask], unique_cells_class[~neuronal_mask_2],
                     unique_cells_cls_color[~neuronal_mask_2], class_folder)

        # Horizontal : 80-140
        # Horizontal : 372-416

        ########################################################################################################################
        # Animation
        ########################################################################################################################

        # import numpy as np
        # import matplotlib.pyplot as plt
        # from matplotlib.animation import FuncAnimation
        #
        #
        # def normalize_pixels(pixel_values, x, y, i, j):
        #     # Ensure x is not equal to y to avoid division by zero
        #     if x == y:
        #         raise ValueError("x and y cannot be the same value")
        #     # Normalize the original values to 0-1, then scale and shift to i-j
        #     new_pixel_values = ((pixel_values - x) / (y - x)) * (j - i) + i
        #     return new_pixel_values
        #
        #
        # def animate(frame):
        #     # Calculate the intermediate coordinates
        #     intermediate_coordinates = initial_coordinates + (final_coordinates - initial_coordinates) * frame / num_frames
        #
        #     # Update the scatter plot
        #     sc.set_offsets(intermediate_coordinates)
        #
        #
        # def flip_array_values(arr):
        #     """
        #     Flip the values of an array such that the min becomes the max and vice versa.
        #
        #     :param arr: Original array.
        #     :return: Flipped array.
        #     """
        #     min_val = np.min(arr)
        #     max_val = np.max(arr)
        #     flipped_arr = max_val + min_val - arr
        #     return flipped_arr
        #
        #
        # i_d = [np.hstack([umap_embeddings["UMAP1"][~neuronal_mask], umap_embeddings["UMAP1"][~neuronal_mask]]),
        #        np.hstack([umap_embeddings["UMAP2"][~neuronal_mask], umap_embeddings["UMAP2"][~neuronal_mask]])]
        # flipped_x_pts = flip_array_values(filtered_points[:, 2][~neuronal_mask]) + 369/2 - np.min(filtered_points[:, 2][~neuronal_mask])
        # #flipped_y_pts = filtered_points[:, 2][~neuronal_mask] + 369
        # f_d = [np.hstack([filtered_points[:, 2][~neuronal_mask], flipped_x_pts]),
        #        np.hstack([filtered_points[:, 0][~neuronal_mask], filtered_points[:, 0][~neuronal_mask]]),
        #        #np.hstack([filtered_points[:, 2][~neuronal_mask], flipped_y_pts]),
        #        ]
        #
        # # Data
        # initial_coordinates = np.dstack([normalize_pixels(i_d[0],
        #                                                   np.min(i_d[0]),
        #                                                   np.max(i_d[0]),
        #                                                   np.min(f_d[0]),
        #                                                   369 - np.min(f_d[0])),
        #                                  normalize_pixels(i_d[1]*0.8,
        #                                                   np.min(i_d[1]),
        #                                                   np.max(i_d[1]),
        #                                                   np.min(-f_d[1]),
        #                                                   np.max(-f_d[1]))])[0]
        # final_coordinates = np.dstack([normalize_pixels(f_d[0],
        #                                                 np.min(f_d[0]),
        #                                                 np.max(f_d[0]),
        #                                                 np.min(f_d[0]),
        #                                                 np.max(f_d[0])),
        #                                normalize_pixels(-f_d[1],
        #                                                 np.min(-f_d[1]),
        #                                                 np.max(-f_d[1]),
        #                                                 np.min(-f_d[1]),
        #                                                 np.max(-f_d[1]))])[0]
        #
        # # Prepare the figure and axis for the animation
        # fig, ax = plt.subplots(figsize=(10, 10))
        # sc = ax.scatter(final_coordinates[:, 0], final_coordinates[:, 1], s=5,
        #                 c="black")
        #                 # c=np.hstack([np.array(cells_cls_color)[~neuronal_mask],
        #                 #             np.array(cells_cls_color)[~neuronal_mask]]),
        #
        # # Setting the axis limits
        # ax.set_xlim(0, 369)
        # ax.set_ylim(-512, 0)
        # ax.set_aspect("equal")
        # ax.set_xticks([])
        # ax.set_yticks([])
        #
        # # Number of frames for the animation
        # num_frames = 1000
        #
        # # Create the animation
        # ani = FuncAnimation(fig, animate, frames=num_frames, interval=20, repeat=False)
        # ani.save(os.path.join(res_dir, f"all_horizontal_class_anim_2.mp4"), writer='ffmpeg', fps=30)
        #
        # plt.show()
        #
        # # Last frame
        # fig = plt.figure()
        # ax = plt.subplot(111)
        # # ax.scatter(final_coordinates[:, 0], final_coordinates[:, 1]+512,
        # #            c=np.hstack([np.array(cells_cls_color)[~neuronal_mask],
        # #                         np.array(cells_cls_color)[~neuronal_mask]]), s=0.1)
        # #
        # ax.scatter(initial_coordinates[:, 0], initial_coordinates[:, 1]+512,
        #            c=np.hstack([np.array(cells_cls_color)[~neuronal_mask],
        #                         np.array(cells_cls_color)[~neuronal_mask]]), s=0.1)
        # ax.imshow(np.rot90(np.max(reference, axis=2)), cmap='gray_r', alpha=0.3)
        # ax.set_xlim(0, 369)
        # ax.set_ylim(0, 512)
        # ax.set_aspect("equal")
        # ax.set_xticks([])
        # ax.set_yticks([])
        # plt.savefig(os.path.join(res_dir, f"neurons_horizontal_class_overlay.png"), dpi=300)
        # plt.show()

        ########################################################################################################################
        #
        ########################################################################################################################
        #
        # # ABA ref: 456, 320, 528
        # # gubra ref: 369, 512, 268
        # reference_file = r"C:\Users\MANMONTALCINI\PycharmProjects\ClearMap2\ClearMap\Resources\Atlas\ABA_25um_reference.tif"
        # # reference_file = r"C:\Users\MANMONTALCINI\PycharmProjects\ClearMap2\ClearMap\Resources\Atlas\gubra_template.tif"
        # reference = tifffile.imread(reference_file)
        # reference_max = np.max(reference, axis=0)
        #
        # cluster_to_display = "09 CNU-LGE GABA"
        # cluster_to_display = "33 Vascular"
        # cluster_to_display = "4277 NTS Phox2b Glut_5"
        #
        # cluster_mask = cell_metadata_views["cluster"] == cluster_to_display
        # x_c = cell_metadata_views["z"][cluster_mask]
        # y_c = cell_metadata_views["y"][cluster_mask]
        # cluster_color = cell_metadata_views["cluster_color"][cluster_mask][0]
        #
        # fig = plt.figure()
        # ax = plt.subplot(111)
        # ax.imshow(np.rot90(reference_max)[::-1], cmap='gray_r', alpha=0.3)
        # ax.scatter(np.array(x_c) * 25 * 1.6, np.array(y_c) * 25 * 1.6, s=0.1, alpha=1, c=cluster_color)
        # ax.set_xlim(0, 528)
        # ax.set_ylim(0, 320)
        # ax.invert_yaxis()
        # plt.show()
    #
    # for m in os.listdir(maps_path):
    #
    #     reference_file = r"C:\Users\MANMONTALCINI\PycharmProjects\ClearMap2\ClearMap\Resources\Atlas\gubra_template.tif"
    #     reference = tifffile.imread(reference_file)
    #
    #     # Horizontal
    #     fig = plt.figure()
    #     ax = plt.subplot(111)
    #     ax.scatter(filtered_points[:, 2][~neuronal_mask], filtered_points[:, 0][~neuronal_mask],
    #                c=np.array(cells_cls_color)[~neuronal_mask], s=2, alpha=0.5)
    #     ax.imshow(np.rot90(np.max(reference, axis=2))[::-1], cmap='gray_r', alpha=0.3)
    #     ax.set_xlim(0, 369)
    #     ax.set_ylim(0, 512)
    #     ax.invert_yaxis()
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     plt.savefig(os.path.join(res_dir, f"neurons_horizontal_class.png"), dpi=300)
    #     plt.show()
