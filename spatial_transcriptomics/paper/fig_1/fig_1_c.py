import os
import json
import requests
import numpy as np
import pandas as pd
import tifffile
import anndata
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut

from spatial_transcriptomics.utils.coordinate_manipulation import filter_points_in_3d_mask

ATLAS_USED = "gubra"
DATASETS = np.arange(1, 6, 1)
N_DATASETS = len(DATASETS)
CATEGORY_NAMES = ["class", "subclass", "supertype", "cluster"]
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]

DOWNLOAD_BASE = r"/default/path"  # PERSONAL
MAP_DIR = fr"/default/path"  # PERSONAL
TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"whole_brain_mask.tif"))
RESULTS_DIR = ut.create_dir(os.path.join(MAP_DIR, "results"))
SLICES_DIR = ut.create_dir(os.path.join(RESULTS_DIR, "thick_slices"))

TRANSFORM_DIR = r"resources/abc_atlas"
REFERENCE_FILE = fr"resources/atlas/{ATLAS_USED}_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)

CHUNK_SIZE = 10

for m, ccat in enumerate(CATEGORY_NAMES):

    chunks_start = np.arange(0, TISSUE_MASK.shape[0], CHUNK_SIZE)
    chunks_end = np.arange(CHUNK_SIZE, TISSUE_MASK.shape[0], CHUNK_SIZE)
    if chunks_end[-1] != TISSUE_MASK.shape[0]:
        chunks_end = np.append(chunks_end, TISSUE_MASK.shape[0])
    n_chunks = len(chunks_start)

    for n, (cs, ce) in enumerate(zip(chunks_start, chunks_end)):

        ut.print_c(f"[INFO] Processing chunk: {cs}:{ce}. {n}/{n_chunks}")
        chunk_mask = TISSUE_MASK.copy()
        chunk_mask[0:cs] = 0
        chunk_mask[ce:] = 0

        filtered_points_plane = []
        filtered_points_colors_plane = []

        for i, dataset_n in enumerate(DATASETS):

            ut.print_c(f"[INFO] Loading data from dataset: {dataset_n}")

            if dataset_n < 5:
                dataset_id = f"Zhuang-ABCA-{dataset_n}"
            else:
                dataset_id = f"MERFISH-C57BL6J-638850"
            url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
            manifest = json.loads(requests.get(url).text)
            metadata = manifest['file_listing'][dataset_id]['metadata']
            metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
            metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']
            expression_matrices = manifest['file_listing'][dataset_id]['expression_matrices']

            # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
            cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
            cell_metadata_file_ccf = os.path.join(DOWNLOAD_BASE, cell_metadata_path_ccf)
            cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
            cell_metadata_ccf.set_index('cell_label', inplace=True)
            cell_labels = cell_metadata_ccf.index
            n_cells_ccf = len(cell_metadata_ccf)

            # Filter out the cells
            transformed_coordinates = np.load(os.path.join(TRANSFORM_DIR, f"all_transformed_cells_{ATLAS_USED}_{dataset_n}.npy"))

            # Views
            cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
            cell_metadata_file_views = os.path.join(DOWNLOAD_BASE, cell_metadata_path_views)
            cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
            cell_metadata_views.set_index('cell_label', inplace=True)

            # Filter points
            filtered_points, mask_point = filter_points_in_3d_mask(transformed_coordinates, chunk_mask)
            filtered_labels = np.array(cell_labels)[::-1][mask_point]
            filtered_metadata_views = cell_metadata_views.loc[filtered_labels]

            # Extract data for each category
            cell_classes = filtered_metadata_views["class"].tolist()
            cell_categories = filtered_metadata_views[ccat].tolist()
            cell_colors = filtered_metadata_views[f"{ccat}_color"].tolist()

            non_neuronal_mask = np.array(
                [True if any([j in i for j in NON_NEURONAL_CELL_TYPES]) else False for i in cell_classes])

            if filtered_points.size > 0:
                filtered_points_plane.append(filtered_points[~non_neuronal_mask])
            else:
                filtered_points_plane.append(np.array([]))

            if cell_colors:
                filtered_points_colors_plane.append(np.array(cell_colors)[~non_neuronal_mask])
            else:
                filtered_points_colors_plane.append(np.array([]))

        ############################################################################################################
        # SAVE FLYTHROUGH IN CORONAL
        ############################################################################################################

        # Filter out empty arrays
        non_empty_arrays = [arr for arr in filtered_points_plane if arr.size > 0]
        if non_empty_arrays:
            filtered_points_plane = np.concatenate(non_empty_arrays)
            sorted_z_indices = np.argsort(filtered_points_plane[:, 0])
        else:
            filtered_points_plane = np.array([])
            sorted_z_indices = np.array([])
        filtered_points_colors_plane = np.concatenate(filtered_points_colors_plane)

        ut.print_c(f"[INFO] plotting {len(filtered_points_plane)} cells")

        fig = plt.figure()
        ax = plt.subplot(111)

        if filtered_points_plane.size > 0:
            ax.scatter(filtered_points_plane[:, 2][sorted_z_indices],
                       filtered_points_plane[:, 1][sorted_z_indices],
                       c=filtered_points_colors_plane[sorted_z_indices], s=1,
                       lw=0., edgecolors="black", alpha=1)

        reference_plane = (cs + ce) / 2
        ax.imshow(np.rot90(np.max(REFERENCE[:, cs:ce, :], 1))[::-1], cmap='gray_r', alpha=0.3)
        ax.set_xlim(0, 369)
        ax.set_ylim(0, 268)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(os.path.join(SLICES_DIR, f"{ccat}_{cs}-{ce}.png"), dpi=300)
        plt.close()
