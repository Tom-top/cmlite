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
import matplotlib
import time

matplotlib.use("Agg")

import utils.utils as ut


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


def filter_points_in_3d_mask(arr_0, mask_1, verbose=False):
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
    if verbose:
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


def setup_plot(n, i):
    if i is None:
        if n == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            return fig, ax
    else:
        if i == 0 and n == 0:
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
            if type(cell_colors) != str:
                if cell_colors.size > 0:
                    cell_colors = cell_colors[~neuronal_mask]
        ax.scatter(filtered_points_plot_x,
                   filtered_points_plot_y,
                   c=cell_colors,
                   s=kwargs["marker_size"],
                   lw=kwargs["linewidth"], edgecolors="black", alpha=1)
    if i is None:
        if n + 1 == kwargs["n_chunks"]:
            ax.imshow(np.rot90(np.max(kwargs["reference"], axis=orip))[::-1], cmap='gray_r', alpha=0.3)
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)
            ax.invert_yaxis()
            ax.axis('off')
            fig.savefig(os.path.join(kwargs["saving_dir"], saving_name), dpi=300)
    else:
        if i + 1 == kwargs["n_datasets"] and n + 1 == kwargs["n_chunks"]:
            ax.imshow(np.rot90(np.max(kwargs["reference"], axis=orip))[::-1], cmap='gray_r', alpha=0.3)
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)
            ax.invert_yaxis()
            ax.axis('off')
            fig.savefig(os.path.join(kwargs["saving_dir"], saving_name), dpi=300)


datasets = [1, 2, 3, 4]
n_datasets = len(datasets)

category_names = ["class", "subclass", "supertype", "cluster"]
#category_names = ["subclass"]

# Get unique categories
dataset_id = f"Zhuang-ABCA-1"
DOWNLOAD_BASE = r'E:\tto\spatial_transcriptomics'
url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
manifest = json.loads(requests.get(url).text)
metadata = manifest['file_listing'][dataset_id]['metadata']
metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
cell_metadata_file_views = os.path.join(DOWNLOAD_BASE, cell_metadata_path_views)
cell_metadata_views_o = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
cell_metadata_views_o.set_index('cell_label', inplace=True)
# category_unique_values = [np.unique(cell_metadata_views_o[x]) for x in category_names]
category_unique_values = [cell_metadata_views_o.drop_duplicates(subset=x)[x].tolist() for x in category_names]
category_unique_colors = [cell_metadata_views_o.drop_duplicates(subset=x)[f"{x}_color"].tolist() for x in category_names]

for m, (ccat, cat_vs, cat_cs) in enumerate(zip(category_names, category_unique_values, category_unique_colors)):

    print(f"Generating figures for category: {ccat}")

    for cat_v, cat_c in zip(cat_vs, cat_cs):

        cat_available = False
        if ccat == "class":
            if np.sum(cell_metadata_views_o["class"] == cat_v) > 0:
                cat_available = True
        elif ccat == "subclass":
            if np.sum(cell_metadata_views_o["subclass"] == cat_v) > 0:
                cat_available = True
                class_cat = cell_metadata_views_o["class"][cell_metadata_views_o["subclass"] == cat_v][0]
        elif ccat == "supertype":
            if np.sum(cell_metadata_views_o["supertype"] == cat_v) > 0:
                cat_available = True
                class_cat = cell_metadata_views_o["class"][cell_metadata_views_o["supertype"] == cat_v][0]
                subclass_cat = cell_metadata_views_o["subclass"][cell_metadata_views_o["supertype"] == cat_v][0]
        elif ccat == "cluster":
            if np.sum(cell_metadata_views_o["cluster"] == cat_v) > 0:
                cat_available = True
                class_cat = cell_metadata_views_o["class"][cell_metadata_views_o["cluster"] == cat_v][0]
                subclass_cat = cell_metadata_views_o["subclass"][cell_metadata_views_o["cluster"] == cat_v][0]
                supertype_cat = cell_metadata_views_o["supertype"][cell_metadata_views_o["cluster"] == cat_v][0]

        if cat_available:

            print(f"Generating figures for category: {ccat}-{cat_v}")

            for i, dataset_n in enumerate(datasets):

                print(f"Loading data from mouse {dataset_n}")

                dataset_id = f"Zhuang-ABCA-{dataset_n}"
                url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
                manifest = json.loads(requests.get(url).text)
                metadata = manifest['file_listing'][dataset_id]['metadata']
                metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
                metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']
                expression_matrices = manifest['file_listing'][dataset_id]['expression_matrices']
                cell_metadata_path = expression_matrices[dataset_id]['log2']['files']['h5ad']['relative_path']
                file = os.path.join(DOWNLOAD_BASE, cell_metadata_path)
                adata = anndata.read_h5ad(file, backed='r')
                genes = adata.var

                # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
                cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
                cell_metadata_file_ccf = os.path.join(DOWNLOAD_BASE, cell_metadata_path_ccf)
                cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
                cell_metadata_ccf.set_index('cell_label', inplace=True)
                cell_labels = cell_metadata_ccf.index
                n_cells_ccf = len(cell_metadata_ccf)

                # Filter out the cells
                transformed_coordinates = np.load(os.path.join(DOWNLOAD_BASE,
                                                               rf"E:\tto\spatial_transcriptomics\transformed_cells\all_transformed_cells_{dataset_n}.npy"))

                RESULTS_DIR = os.path.join(DOWNLOAD_BASE, "results")
                WHOLE_BRAIN_RESULTS_DIR = ut.create_dir(os.path.join(RESULTS_DIR, "whole_brain"))
                RES_DIR = ut.create_dir(os.path.join(WHOLE_BRAIN_RESULTS_DIR, "categories"))
                map = "whole_brain"

                # mask = tifffile.imread(os.path.join(map_path, r"bin.tif"))
                mask = tifffile.imread(os.path.join(WHOLE_BRAIN_RESULTS_DIR, r"bin_whole.tif"))

                chunk_size = 10
                # overlap_size = 0
                chunks_start = np.arange(0, mask.shape[0], chunk_size)
                # chunks_end = np.arange(chunk_size + overlap_size, mask.shape[0], chunk_size)
                chunks_end = np.arange(chunk_size, mask.shape[0], chunk_size)
                if chunks_end[-1] != mask.shape[0]:
                    chunks_end = np.append(chunks_end, mask.shape[0])
                n_chunks = len(chunks_start)

                # Views
                cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
                cell_metadata_file_views = os.path.join(DOWNLOAD_BASE, cell_metadata_path_views)
                cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
                cell_metadata_views.set_index('cell_label', inplace=True)

                # gubra ref: 369, 512, 268
                reference_file = r"C:\Users\MANDOUDNA\PycharmProjects\cmlite\resources\atlas\gubra_reference_mouse.tif"
                reference = tifffile.imread(reference_file)

                for n, (cs, ce) in enumerate(zip(chunks_start, chunks_end)):

                    chunk_mask = mask.copy()
                    chunk_mask[0:cs] = 0
                    chunk_mask[ce:] = 0

                    #print(f"Processing chunk: {cs}:{ce}. {n}/{n_chunks}")
                    #print("Filtering points in mask!")
                    filtered_points, mask_point = filter_points_in_3d_mask(transformed_coordinates, chunk_mask,
                                                                           verbose=False)
                    #filtered_points = transformed_coordinates
                    filtered_labels = cell_labels[::-1][mask_point]
                    #filtered_labels = cell_labels[::-1]

                    # Get the relevant rows from cell_metadata_views in one operation
                    # print("Filtering cell metadata views!")
                    filtered_metadata_views = cell_metadata_views.loc[filtered_labels]
                    cat_mask = filtered_metadata_views[f"{ccat}"] == cat_v
                    #filtered_metadata_views = filtered_metadata_views[filtered_metadata_views[f"{ccat}"] == cat_v]

                    filtered_points = filtered_points[cat_mask]

                    ########################################################################################################################
                    # Color transformed points
                    ########################################################################################################################

                    plot_cells_params = {
                        "n_datasets": n_datasets,
                        "reference": reference,
                        "saving_dir": RES_DIR,
                        "filtered_points": filtered_points,
                        "n_chunks": n_chunks,
                        "marker_size": 0.1,
                        "linewidth": 0.,
                    }

                    if ccat == "class":
                        saving_path = f"{cat_v.replace('/', '-')}"
                    elif ccat == "subclass":
                        saving_path = f"{class_cat.replace('/', '-')}/{cat_v.replace('/', '-')}"
                    elif ccat == "supertype":
                        saving_path = (f"{class_cat.replace('/', '-')}/"
                                       f"{subclass_cat.replace('/', '-')}/{cat_v.replace('/', '-')}")
                    elif ccat == "cluster":
                        saving_path = (f"{class_cat.replace('/', '-')}/"
                                       f"{subclass_cat.replace('/', '-')}/{supertype_cat.replace('/', '-')}/"
                                       f"{cat_v.replace('/', '-')}")
                    if not os.path.exists(os.path.join(RES_DIR, saving_path)):
                        os.mkdir(os.path.join(RES_DIR, saving_path))

                    ############################################################################################################
                    # Horizontal
                    ############################################################################################################

                    ori = "horizontal"
                    orix, oriy = 2, 0
                    xlim, ylim = 369, 512

                    # Eighth horizontal plot: All cells, class colors, all experiments
                    if i == 0 and n == 0:
                        fig1abc, ax1abc = setup_plot(n, i)
                    plot_cells(n, i, fig1abc, ax1abc, cell_colors=cat_c, neuronal_mask=None,
                               xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                               saving_name=f"{saving_path}/{ccat}_{cat_v.replace('/', '-')}_{ori}.png", **plot_cells_params)

                    ############################################################################################################
                    # Sagittal
                    ############################################################################################################

                    ori = "sagittal"
                    orix, oriy = 0, 1
                    xlim, ylim = 512, 268

                    # Eighth sagittal plot: All cells, class colors, all experiments
                    if i == 0 and n == 0:
                        fig2abc, ax2abc = setup_plot(n, i)
                    plot_cells(n, i, fig2abc, ax2abc, cell_colors=cat_c, neuronal_mask=None,
                               xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                               saving_name=f"{saving_path}/{ccat}_{cat_v.replace('/', '-')}_{ori}.png", **plot_cells_params)

                    ############################################################################################################
                    # Coronal
                    ############################################################################################################

                    ori = "coronal"
                    orix, oriy = 2, 1 # Projection = 1
                    xlim, ylim = 369, 268

                    # Eighth coronal plot: All cells, class colors, all experiments
                    if i == 0 and n == 0:
                        fig3abc, ax3abc = setup_plot(n, i)
                    plot_cells(n, i, fig3abc, ax3abc, cell_colors=cat_c, neuronal_mask=None,
                               xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                               saving_name=f"{saving_path}/{ccat}_{cat_v.replace('/', '-')}_{ori}.png", **plot_cells_params)