"""
This script generates visualizations showing the distribution of cells across four individual animals
 as well as a combined dataset.
For each dataset, coronal, sagittal, and horizontal views are produced with the Gubra LSFM reference in the background.
The cells are labeled according to the ABC atlas ontology (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas).
The source of the data is from Zhang M. et al., Nature, 2023 (DOI: 10.1038/s41586-023-06808-9) (datasets 1-4)
and Yao Z. et al., Nature, 2023 (DOI: 10.1038/s41586-023-06812-z) (dataset 5).

Author: Thomas Topilko
"""


import os
import json
import requests
import numpy as np
import pandas as pd
import tifffile
import anndata
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut

import spatial_transcriptomics.utils.utils as sut
from spatial_transcriptomics.utils.coordinate_manipulation import filter_points_in_3d_mask
from spatial_transcriptomics.utils.plotting import setup_plot, plot_cells


########################################################################################################################
# SET GLOBAL VARIABLES
########################################################################################################################

ATLAS_USED = "gubra"
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]

########################################################################################################################
# SET PATHS
########################################################################################################################

DOWNLOAD_BASE = r'E:\tto\spatial_transcriptomics'  # PERSONAL
URL = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
MANIFEST = sut.fetch_manifest(URL)
DATASETS = np.arange(1, 6, 1)
N_DATASETS = len(DATASETS)
DATASET_COLORS = ["cyan", "magenta", "yellow", "black", "orange"]
CATEGORY_NAMES = ["class", "subclass", "supertype", "cluster"]

RESOURCES_DIR = "resources"
ATLAS_DIR = os.path.join(RESOURCES_DIR, "atlas")
REFERENCE_FILE = os.path.join(ATLAS_DIR, f"{ATLAS_USED}_reference_mouse.tif")
REFERENCE = tifffile.imread(REFERENCE_FILE)
REFERENCE_SHAPE = REFERENCE.shape
TRANSFORM_DIR = os.path.join(RESOURCES_DIR, "abc_atlas")
TRANSFORM_DIR = r"E:\tto\mapping_aba_to_gubra"

MAP_DIR = ut.create_dir(r"/default/path")  # PERSONAL
TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"whole_brain_mask.tif"))
WORKING_DIRECTORY = ut.create_dir(os.path.join(MAP_DIR, r"results"))
SAVING_DIRECTORY = ut.create_dir(os.path.join(WORKING_DIRECTORY, f"{ATLAS_USED}"))

for m, ccat in enumerate(CATEGORY_NAMES):

    # cats = []
    # cats_neurons = []
    # unique_classes = []

    for i, (dataset_n, dataset_c) in enumerate(zip(DATASETS, DATASET_COLORS)):

        SAVING_DIR = ut.create_dir(os.path.join(SAVING_DIRECTORY, f"mouse_{dataset_n}"))

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
        if dataset_n < 5:
            cell_metadata_path = expression_matrices[dataset_id]['log2']['files']['h5ad']['relative_path']
        else:
            cell_metadata_path = expression_matrices["-".join(dataset_id.split("-")[1:])]['log2']['files']['h5ad'][
                'relative_path']
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
        transformed_coordinates = np.load(os.path.join(TRANSFORM_DIR, f"all_transformed_cells_{ATLAS_USED}_{dataset_n}.npy"))

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

        # n_cells = 0
        # n_neurons = 0

        for n, (cs, ce) in enumerate(zip(chunks_start, chunks_end)):

            chunk_mask = TISSUE_MASK.copy()
            chunk_mask[0:cs] = 0
            chunk_mask[ce:] = 0

            # print(f"Processing chunk: {cs}:{ce}. {n}/{n_chunks}")
            filtered_points, mask_point = filter_points_in_3d_mask(transformed_coordinates, chunk_mask)
            filtered_labels = np.array(cell_labels)[::-1][mask_point]
            filtered_metadata_views = cell_metadata_views.loc[filtered_labels]

            # Extract data for each category
            cells_class = filtered_metadata_views["class"].tolist()
            cells_class_color = filtered_metadata_views["class_color"].tolist()
            cells_subclass = filtered_metadata_views["subclass"].tolist()
            cells_subclass_color = filtered_metadata_views["subclass_color"].tolist()
            cells_supertype = filtered_metadata_views["supertype"].tolist()
            cells_supertype_color = filtered_metadata_views["supertype_color"].tolist()
            cells_cluster = filtered_metadata_views["cluster"].tolist()
            cells_cluster_color = filtered_metadata_views["cluster_color"].tolist()

            non_neuronal_mask = np.array(
                [True if any([j in i for j in NON_NEURONAL_CELL_TYPES]) else False for i in cells_class])

            unique_cells_class, unique_indices = np.unique(cells_class, return_index=True)
            unique_cells_class_color = np.array(cells_class_color)[unique_indices]
            if len(cells_class) > 0:
                unique_neurons_class, unique_indices = np.unique(np.array(cells_class)[~non_neuronal_mask], return_index=True)
                unique_neurons_class_color = np.array(cells_class_color)[~non_neuronal_mask][unique_indices]
            else:
                unique_neurons_class, unique_neurons_class_color = np.array([]), np.array([])

            unique_cells_subclass, unique_indices = np.unique(cells_subclass, return_index=True)
            unique_cells_subclass_color = np.array(cells_subclass_color)[unique_indices]
            if len(cells_subclass) > 0:
                unique_neurons_subclass, unique_indices = np.unique(np.array(cells_subclass)[~non_neuronal_mask], return_index=True)
                unique_neurons_subclass_color = np.array(cells_subclass_color)[~non_neuronal_mask][unique_indices]
            else:
                unique_neurons_subclass, unique_neurons_subclass_color = np.array([]), np.array([])

            unique_cells_supertype, unique_indices = np.unique(cells_supertype, return_index=True)
            unique_cells_supertype_color = np.array(cells_supertype_color)[unique_indices]
            if len(cells_supertype) > 0:
                unique_neurons_supertype, unique_indices = np.unique(np.array(cells_supertype)[~non_neuronal_mask], return_index=True)
                unique_neurons_supertype_color = np.array(cells_supertype_color)[~non_neuronal_mask][unique_indices]
            else:
                unique_neurons_supertype, unique_neurons_supertype_color = np.array([]), np.array([])

            unique_cells_cluster, unique_indices = np.unique(cells_cluster, return_index=True)
            unique_cells_cluster_color = np.array(cells_cluster_color)[unique_indices]
            if len(cells_cluster) > 0:
                unique_neurons_cluster, unique_indices = np.unique(np.array(cells_cluster)[~non_neuronal_mask], return_index=True)
                unique_neurons_cluster_color = np.array(cells_cluster_color)[~non_neuronal_mask][unique_indices]
            else:
                unique_neurons_cluster, unique_neurons_cluster_color = np.array([]), np.array([])

    #         selected_cat = "cluster"
    #         n_cells += len(filtered_points)
    #         if not non_neuronal_mask.size == 0:
    #             n_neurons += np.sum(~non_neuronal_mask)
    #         cat_data = globals()[f"unique_cells_{selected_cat}"]
    #         cats.append(cat_data)
    #         if cat_data.size == 0:
    #             cats_neurons.append(np.array([]))
    #         else:
    #             cats_neurons.append(globals()[f"unique_neurons_{selected_cat}"])
    #
    #     print(f"{n_cells} cells detected in dataset {i}")
    #     print(f"{n_neurons} neurons detected in dataset {i}")
    # unique_cats = np.unique(np.concatenate(cats).ravel())
    # unique_cats_neurons = np.unique(np.concatenate(cats_neurons).ravel())
    # print(f"{len(unique_cats)} unique cats")
    # print(f"{len(unique_cats_neurons)} unique cats neurons")
    # break


    # def find_missing_numbers(lst, X):
    #     # Create a set of all numbers from 1 to X
    #     full_set = set(range(1, X + 1))
    #
    #     # Convert the list to a set and find the difference
    #     missing_numbers = full_set - set(lst)
    #
    #     # Convert the result back to a sorted list
    #     return sorted(list(missing_numbers))
    #
    #
    # numbers = [int(i[:4]) for i in unique_cats]
    # missing_numbers = find_missing_numbers(numbers, 1201)


        if ccat == "class":
            cc = cells_class_color
        elif ccat == "subclass":
            cc = cells_subclass_color
        elif ccat == "supertype":
            cc = cells_supertype_color
        elif ccat == "cluster":
            cc = cells_cluster_color

        ########################################################################################################################
        # Color transformed points
        ########################################################################################################################

        plot_cells_params = {
            "n_datasets": N_DATASETS,
            "reference": REFERENCE,
            "saving_dir": SAVING_DIR,
            "filtered_points": filtered_points,
            "n_chunks": n_chunks,
            "marker_size": 0.1,
            "linewidth": 0.,
        }

        all_plots_cells_params = plot_cells_params.copy()
        all_plots_cells_params["saving_dir"] = SAVING_DIRECTORY

        # marker_size = 0.1
        # linewidth = 0.1

        if m == 0:

            ############################################################################################################
            # Horizontal
            ############################################################################################################

            ori = "horizontal"
            orix, oriy = 2, 0
            xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[1]

            # First horizontal plot: Only neurons, monochrome, current experiment
            if n == 0:
                fig1, ax1 = setup_plot(n, None)
            plot_cells(n, None, fig1, ax1, cell_colors=dataset_c, non_neuronal_mask=non_neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_path=f"neurons_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

            # Third horizontal plot: Only neurons, monochrome, all experiments
            if i == 0 and n == 0:
                fig1b, ax1b = setup_plot(n, i)
            plot_cells(n, i, fig1b, ax1b, cell_colors=dataset_c, non_neuronal_mask=non_neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_path=f"neurons_{ori}_mouse.png", **all_plots_cells_params)

            # Fifth horizontal plot: All cells, monochrome, current experiment
            if n == 0:
                fig1a, ax1a = setup_plot(n, None)
            plot_cells(n, None, fig1a, ax1a, cell_colors=dataset_c, non_neuronal_mask=None,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_path=f"all_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

            # Seventh horizontal plot: All cells, monochrome, all experiments
            if i == 0 and n == 0:
                fig1ab, ax1ab = setup_plot(n, i)
            plot_cells(n, i, fig1ab, ax1ab, cell_colors=dataset_c, non_neuronal_mask=None,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_path=f"all_{ori}_mouse.png", **all_plots_cells_params)

            ############################################################################################################
            # Sagittal
            ############################################################################################################

            ori = "sagittal"
            orix, oriy = 0, 1
            xlim, ylim = REFERENCE_SHAPE[1], REFERENCE_SHAPE[2]

            # First sagittal plot: Only neurons, monochrome, current experiment
            if n == 0:
                fig2, ax2 = setup_plot(n, None)
            plot_cells(n, None, fig2, ax2, cell_colors=dataset_c, non_neuronal_mask=non_neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_path=f"neurons_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

            # Third sagittal plot: Only neurons, monochrome, all experiments
            if i == 0 and n == 0:
                fig2b, ax2b = setup_plot(n, i)
            plot_cells(n, i, fig2b, ax2b, cell_colors=dataset_c, non_neuronal_mask=non_neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_path=f"neurons_{ori}_mouse.png", **all_plots_cells_params)

            # Fifth sagittal plot: All cells, monochrome, current experiment
            if n == 0:
                fig2a, ax2a = setup_plot(n, None)
            plot_cells(n, None, fig2a, ax2a, cell_colors=dataset_c, non_neuronal_mask=None,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_path=f"all_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

            # Seventh sagittal plot: All cells, monochrome, all experiments
            if i == 0 and n == 0:
                fig2ab, ax2ab = setup_plot(n, i)
            plot_cells(n, i, fig2ab, ax2ab, cell_colors=dataset_c, non_neuronal_mask=None,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_path=f"all_{ori}_mouse.png", **all_plots_cells_params)

            ############################################################################################################
            # Coronal
            ############################################################################################################

            ori = "coronal"
            orix, oriy = 2, 1  # Projection = 1
            xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[2]

            # First coronal plot: Only neurons, monochrome, current experiment
            if n == 0:
                fig3, ax3 = setup_plot(n, None)
            plot_cells(n, None, fig3, ax3, cell_colors=dataset_c, non_neuronal_mask=non_neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                       saving_path=f"neurons_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

            # Third coronal plot: Only neurons, monochrome, all experiments
            if i == 0 and n == 0:
                fig3b, ax3b = setup_plot(n, i)
            plot_cells(n, i, fig3b, ax3b, cell_colors=dataset_c, non_neuronal_mask=non_neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                       saving_name=f"neurons_{ori}_mouse.png", **all_plots_cells_params)

            # Fifth coronal plot: All cells, monochrome, current experiment
            if n == 0:
                fig3a, ax3a = setup_plot(n, None)
            plot_cells(n, None, fig3a, ax3a, cell_colors=dataset_c, non_neuronal_mask=None,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                       saving_name=f"all_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

            # Seventh coronal plot: All cells, monochrome, all experiments
            if i == 0 and n == 0:
                fig3ab, ax3ab = setup_plot(n, i)
            plot_cells(n, i, fig3ab, ax3ab, cell_colors=dataset_c, non_neuronal_mask=None,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                       saving_name=f"all_{ori}_mouse.png", **all_plots_cells_params)

        ############################################################################################################
        # Horizontal
        ############################################################################################################

        ori = "horizontal"
        orix, oriy = 2, 0
        xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[1]

        # Second horizontal plot: Only neurons, class colors, current experiment
        if n == 0:
            fig1c, ax1c = setup_plot(n, None)
        plot_cells(n, None, fig1c, ax1c, cell_colors=np.array(cc), non_neuronal_mask=non_neuronal_mask,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                   saving_path=f"neurons_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

        # Forth horizontal plot: Only neurons, class colors, all experiments
        if i == 0 and n == 0:
            fig1bc, ax1bc = setup_plot(n, i)
        plot_cells(n, i, fig1bc, ax1bc, cell_colors=np.array(cc), non_neuronal_mask=non_neuronal_mask,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                   saving_path=f"neurons_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

        # Sixth horizontal plot: All cells, class colors, current experiment
        if n == 0:
            fig1ac, ax1ac = setup_plot(n, None)
        plot_cells(n, None, fig1ac, ax1ac, cell_colors=np.array(cc), non_neuronal_mask=None,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                   saving_path=f"all_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

        # Eighth horizontal plot: All cells, class colors, all experiments
        if i == 0 and n == 0:
            fig1abc, ax1abc = setup_plot(n, i)
        plot_cells(n, i, fig1abc, ax1abc, cell_colors=np.array(cc), non_neuronal_mask=None,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                   saving_path=f"all_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

        ############################################################################################################
        # Sagittal
        ############################################################################################################

        ori = "sagittal"
        orix, oriy = 0, 1
        xlim, ylim = REFERENCE_SHAPE[1], REFERENCE_SHAPE[2]

        # Second sagittal plot: Only neurons, class colors, current experiment
        if n == 0:
            fig2c, ax2c = setup_plot(n, None)
        plot_cells(n, None, fig2c, ax2c, cell_colors=np.array(cc), non_neuronal_mask=non_neuronal_mask,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                   saving_path=f"neurons_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

        # Forth sagittal plot: Only neurons, class colors, all experiments
        if i == 0 and n == 0:
            fig2bc, ax2bc = setup_plot(n, i)
        plot_cells(n, i, fig2bc, ax2bc, cell_colors=np.array(cc), non_neuronal_mask=non_neuronal_mask,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                   saving_path=f"neurons_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

        # Sixth sagittal plot: All cells, class colors, current experiment
        if n == 0:
            fig2ac, ax2ac = setup_plot(n, None)
        plot_cells(n, None, fig2ac, ax2ac, cell_colors=np.array(cc), non_neuronal_mask=None,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                   saving_path=f"all_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

        # Eighth sagittal plot: All cells, class colors, all experiments
        if i == 0 and n == 0:
            fig2abc, ax2abc = setup_plot(n, i)
        plot_cells(n, i, fig2abc, ax2abc, cell_colors=np.array(cc), non_neuronal_mask=None,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                   saving_path=f"all_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

        ############################################################################################################
        # Coronal
        ############################################################################################################

        ori = "coronal"
        orix, oriy = 2, 1 # Projection = 1
        xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[2]

        # Second coronal plot: Only neurons, class colors, current experiment
        if n == 0:
            fig3c, ax3c = setup_plot(n, None)
        plot_cells(n, None, fig3c, ax3c, cell_colors=np.array(cc), non_neuronal_mask=non_neuronal_mask,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                   saving_path=f"neurons_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

        # Forth coronal plot: Only neurons, class colors, all experiments
        if i == 0 and n == 0:
            fig3bc, ax3bc = setup_plot(n, i)
        plot_cells(n, i, fig3bc, ax3bc, cell_colors=np.array(cc), non_neuronal_mask=non_neuronal_mask,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                   saving_path=f"neurons_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

        # Sixth coronal plot: All cells, class colors, current experiment
        if n == 0:
            fig3ac, ax3ac = setup_plot(n, None)
        plot_cells(n, None, fig3ac, ax3ac, cell_colors=np.array(cc), non_neuronal_mask=None,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                   saving_path=f"all_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

        # Eighth coronal plot: All cells, class colors, all experiments
        if i == 0 and n == 0:
            fig3abc, ax3abc = setup_plot(n, i)
        plot_cells(n, i, fig3abc, ax3abc, cell_colors=np.array(cc), non_neuronal_mask=None,
                   xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                   saving_path=f"all_{ori}_mouse_{ccat}.png", **all_plots_cells_params)
