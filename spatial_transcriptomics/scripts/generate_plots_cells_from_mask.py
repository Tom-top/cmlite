"""
This script generates visualizations showing the distribution of the cells within the combined datasets
 in a restricted binary mask.
For each dataset, coronal, sagittal, and horizontal views are produced with the Gubra LSFM reference in the background.
The cells are labeled according to the ABC atlas ontology (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas).
The source of the data is from Zhang M. et al., in Nature, 2023 (DOI: 10.1038/s41586-023-06808-9).

Author: Thomas Topilko
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import tifffile
from collections import Counter
import matplotlib
matplotlib.use("Agg")
# matplotlib.use("Qt5Agg")

import utils.utils as ut

from spatial_transcriptomics.utils.coordinate_manipulation import filter_points_in_3d_mask
from spatial_transcriptomics.utils.plotting import setup_plot, plot_cells

datasets = [1, 2, 3, 4]
n_datasets = len(datasets)
category_names = ["class", "subclass", "supertype", "cluster"]
BILATERAL = True

ANO_DIRECTORY = r"resources\atlas"
ANO_PATH = os.path.join(ANO_DIRECTORY, "gubra_annotation_mouse.tif")
ANO = np.transpose(tifffile.imread(ANO_PATH), (1, 2, 0))
ANO_JSON = os.path.join(ANO_DIRECTORY, "gubra_annotation_mouse.json")

DOWNLOAD_BASE = r"E:\tto\spatial_transcriptomics"  # PERSONAL
MAP_DIR = r"E:\tto\spatial_transcriptomics_results\Semaglutide"  # PERSONAL
LABELED_MASK = False  # If true the TISSUE_MASK is a labeled 32bit mask, not a binary.
if LABELED_MASK:  # Each label will be processed separately.
    TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"labeled_mask.tif"))
    unique_labels = np.unique(TISSUE_MASK)
    TISSUE_MASKS = [(TISSUE_MASK == ul).astype("uint8")*255 for ul in unique_labels if not ul == 0]
    labels = [ul for ul in unique_labels if not ul == 0]
else:
    labels = [1]
    TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"hemisphere_mask.tif"))
    TISSUE_MASKS = [TISSUE_MASK]

MAIN_RESULTS_DIR = os.path.join(MAP_DIR, "results")
RESULTS_DIR = ut.create_dir(os.path.join(MAIN_RESULTS_DIR, "3d_views"))

TRANSFORM_DIR = r"resources/abc_atlas"
REFERENCE_FILE = r"resources/atlas/gubra_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)

for ul, TISSUE_MASK in zip(labels, TISSUE_MASKS):

    region_id_counts = Counter(ANO[TISSUE_MASKS[0] == 255])
    most_common_region_id = max(region_id_counts, key=region_id_counts.get)
    structure = ut.read_ano_json(ANO_JSON)
    region_acronym = ut.find_acronym_by_id(structure, most_common_region_id)

    with open(ANO_JSON, 'r') as f:
        json_data = json.load(f)
    structure = json_data['msg'][0]

    if LABELED_MASK:
        SAVING_DIR = os.path.join(RESULTS_DIR, f"{region_acronym}_blob_1")
        if os.path.exists(SAVING_DIR):
            n_region_dirs = len([n for n in os.listdir(RESULTS_DIR) if n.startswith(region_acronym)])
            SAVING_DIR = os.path.join(RESULTS_DIR, f"{region_acronym}_blob_{n_region_dirs+1}")
        SAVING_DIR = ut.create_dir(SAVING_DIR)
    else:
        SAVING_DIR = RESULTS_DIR

    for m, ccat in enumerate(category_names):

        for i, dataset_n in enumerate(datasets):

            dataset_id = f"Zhuang-ABCA-{dataset_n}"
            url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
            manifest = json.loads(requests.get(url).text)
            metadata = manifest['file_listing'][dataset_id]['metadata']
            metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
            metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']

            # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
            cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
            cell_metadata_file_ccf = os.path.join(DOWNLOAD_BASE, cell_metadata_path_ccf)
            cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
            cell_metadata_ccf.set_index('cell_label', inplace=True)
            cell_labels = cell_metadata_ccf.index

            # Filter out the cells
            transformed_coordinates = np.load(os.path.join(TRANSFORM_DIR, f"all_transformed_cells_{dataset_n}.npy"))

            chunk_size = 10
            chunks_start = np.arange(0, TISSUE_MASK.shape[0], chunk_size)
            chunks_end = np.arange(chunk_size, TISSUE_MASK.shape[0], chunk_size)
            if chunks_end[-1] != TISSUE_MASK.shape[0]:
                chunks_end = np.append(chunks_end, TISSUE_MASK.shape[0])
            n_chunks = len(chunks_start)

            # Views
            cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
            cell_metadata_file_views = os.path.join(DOWNLOAD_BASE, cell_metadata_path_views)
            cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
            cell_metadata_views.set_index('cell_label', inplace=True)

            for n, (cs, ce) in enumerate(zip(chunks_start, chunks_end)):

                chunk_mask = TISSUE_MASK.copy()
                chunk_mask[0:cs] = 0
                chunk_mask[ce:] = 0

                print(f"Processing chunk: {cs}:{ce}. {n}/{n_chunks}")
                filtered_points, mask_point = filter_points_in_3d_mask(transformed_coordinates, chunk_mask)
                filtered_labels = np.array(cell_labels)[::-1][mask_point]
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

                if ccat == "class":
                    cc = cells_cls_color
                elif ccat == "subclass":
                    cc = cells_subcls_color
                elif ccat == "supertype":
                    cc = cells_supertype_color
                elif ccat == "cluster":
                    cc = cells_cluster_color

                ########################################################################################################################
                # Color transformed points
                ########################################################################################################################

                if BILATERAL:
                    mirrored_filtered_points = filtered_points.copy()
                    mirrored_filtered_points[:, 2] = REFERENCE.shape[0] - 1 - filtered_points[:, 2]
                    filtered_points = np.concatenate([filtered_points, mirrored_filtered_points])
                    neuronal_mask = np.tile(neuronal_mask, 2)
                    cc = np.tile(np.array(cc), 2)

                all_plots_cells_params = {
                    "n_datasets": n_datasets,
                    "reference": REFERENCE,
                    "saving_dir": SAVING_DIR,
                    "filtered_points": filtered_points,
                    "n_chunks": n_chunks,
                    "marker_size": 0.5,
                    "linewidth": 0.,
                }

                ############################################################################################################
                # Horizontal
                ############################################################################################################

                ori = "horizontal"
                orix, oriy = 2, 0
                xlim, ylim = 369, 512

                # Forth horizontal plot: Only neurons, class colors, all experiments
                if i == 0 and n == 0:
                    fig1bc, ax1bc = setup_plot(n, i)
                plot_cells(n, i, fig1bc, ax1bc, cell_colors=np.array(cc), neuronal_mask=neuronal_mask,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"neurons_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

                # Eighth horizontal plot: All cells, class colors, all experiments
                if i == 0 and n == 0:
                    fig1abc, ax1abc = setup_plot(n, i)
                plot_cells(n, i, fig1abc, ax1abc, cell_colors=np.array(cc), neuronal_mask=None,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"all_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

                ############################################################################################################
                # Sagittal
                ############################################################################################################

                ori = "sagittal"
                orix, oriy = 0, 1
                xlim, ylim = 512, 268

                # Forth sagittal plot: Only neurons, class colors, all experiments
                if i == 0 and n == 0:
                    fig2bc, ax2bc = setup_plot(n, i)
                plot_cells(n, i, fig2bc, ax2bc, cell_colors=np.array(cc), neuronal_mask=neuronal_mask,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"neurons_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

                # Eighth sagittal plot: All cells, class colors, all experiments
                if i == 0 and n == 0:
                    fig2abc, ax2abc = setup_plot(n, i)
                plot_cells(n, i, fig2abc, ax2abc, cell_colors=np.array(cc), neuronal_mask=None,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"all_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

                ############################################################################################################
                # Coronal
                ############################################################################################################

                ori = "coronal"
                orix, oriy = 2, 1 # Projection = 1
                xlim, ylim = 369, 268

                # Forth coronal plot: Only neurons, class colors, all experiments
                if i == 0 and n == 0:
                    fig3bc, ax3bc = setup_plot(n, i)
                plot_cells(n, i, fig3bc, ax3bc, cell_colors=np.array(cc), neuronal_mask=neuronal_mask,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                           saving_name=f"neurons_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

                # Eighth coronal plot: All cells, class colors, all experiments
                if i == 0 and n == 0:
                    fig3abc, ax3abc = setup_plot(n, i)
                plot_cells(n, i, fig3abc, ax3abc, cell_colors=np.array(cc), neuronal_mask=None,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                           saving_name=f"all_{ori}_mouse_{ccat}.png", **all_plots_cells_params)