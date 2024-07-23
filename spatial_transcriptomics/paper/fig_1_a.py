"""
This script generates visualizations showing the distribution of cells across four individual animals
 as well as a combined dataset.
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
import anndata
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut

from spatial_transcriptomics.utils.coordinate_manipulation import filter_points_in_3d_mask
from spatial_transcriptomics.utils.plotting import setup_plot, plot_cells

datasets = [1, 2, 3, 4]
dataset_colors = ["cyan", "magenta", "yellow", "black"]
n_datasets = len(datasets)
category_names = ["class", "subclass", "supertype", "cluster"]

DOWNLOAD_BASE = r"/default/path"  # PERSONAL
MAP_DIR = r"/mnt/data/Thomas/whole_brain"  # PERSONAL
TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"whole_brain_mask.tif"))
RESULTS_DIR = os.path.join(MAP_DIR, "results")

TRANSFORM_DIR = r"resources/abc_atlas"
REFERENCE_FILE = r"resources/atlas/gubra_reference_mouse.tif"

for m, ccat in enumerate(category_names):

    for i, (dataset_n, dataset_c) in enumerate(zip(datasets, dataset_colors)):

        SAVING_DIR = ut.create_dir(os.path.join(RESULTS_DIR, f"mouse_{dataset_n}"))

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

        # gubra ref: 369, 512, 268
        reference = tifffile.imread(REFERENCE_FILE)

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

            plot_cells_params = {
                "n_datasets": n_datasets,
                "reference": reference,
                "saving_dir": SAVING_DIR,
                "filtered_points": filtered_points,
                "n_chunks": n_chunks,
                "marker_size": 0.1,
                "linewidth": 0.,
            }

            all_plots_cells_params = plot_cells_params.copy()
            all_plots_cells_params["saving_dir"] = RESULTS_DIR

            # marker_size = 0.1
            # linewidth = 0.1

            if m == 0:

                ############################################################################################################
                # Horizontal
                ############################################################################################################

                ori = "horizontal"
                orix, oriy = 2, 0
                xlim, ylim = 369, 512

                # First horizontal plot: Only neurons, monochrome, current experiment
                if n == 0:
                    fig1, ax1 = setup_plot(n, None)
                plot_cells(n, None, fig1, ax1, cell_colors=dataset_c, neuronal_mask=neuronal_mask,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"neurons_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

                # Third horizontal plot: Only neurons, monochrome, all experiments
                if i == 0 and n == 0:
                    fig1b, ax1b = setup_plot(n, i)
                plot_cells(n, i, fig1b, ax1b, cell_colors=dataset_c, neuronal_mask=neuronal_mask,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"neurons_{ori}_mouse.png", **all_plots_cells_params)

                # Fifth horizontal plot: All cells, monochrome, current experiment
                if n == 0:
                    fig1a, ax1a = setup_plot(n, None)
                plot_cells(n, None, fig1a, ax1a, cell_colors=dataset_c, neuronal_mask=None,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"all_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

                # Seventh horizontal plot: All cells, monochrome, all experiments
                if i == 0 and n == 0:
                    fig1ab, ax1ab = setup_plot(n, i)
                plot_cells(n, i, fig1ab, ax1ab, cell_colors=dataset_c, neuronal_mask=None,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"all_{ori}_mouse.png", **all_plots_cells_params)

                ############################################################################################################
                # Sagittal
                ############################################################################################################

                ori = "sagittal"
                orix, oriy = 0, 1
                xlim, ylim = 512, 268

                # First sagittal plot: Only neurons, monochrome, current experiment
                if n == 0:
                    fig2, ax2 = setup_plot(n, None)
                plot_cells(n, None, fig2, ax2, cell_colors=dataset_c, neuronal_mask=neuronal_mask,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"neurons_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

                # Third sagittal plot: Only neurons, monochrome, all experiments
                if i == 0 and n == 0:
                    fig2b, ax2b = setup_plot(n, i)
                plot_cells(n, i, fig2b, ax2b, cell_colors=dataset_c, neuronal_mask=neuronal_mask,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"neurons_{ori}_mouse.png", **all_plots_cells_params)

                # Fifth sagittal plot: All cells, monochrome, current experiment
                if n == 0:
                    fig2a, ax2a = setup_plot(n, None)
                plot_cells(n, None, fig2a, ax2a, cell_colors=dataset_c, neuronal_mask=None,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"all_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

                # Seventh sagittal plot: All cells, monochrome, all experiments
                if i == 0 and n == 0:
                    fig2ab, ax2ab = setup_plot(n, i)
                plot_cells(n, i, fig2ab, ax2ab, cell_colors=dataset_c, neuronal_mask=None,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                           saving_name=f"all_{ori}_mouse.png", **all_plots_cells_params)

                ############################################################################################################
                # Coronal
                ############################################################################################################

                ori = "coronal"
                orix, oriy = 2, 1  # Projection = 1
                xlim, ylim = 369, 268

                # First coronal plot: Only neurons, monochrome, current experiment
                if n == 0:
                    fig3, ax3 = setup_plot(n, None)
                plot_cells(n, None, fig3, ax3, cell_colors=dataset_c, neuronal_mask=neuronal_mask,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                           saving_name=f"neurons_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

                # Third coronal plot: Only neurons, monochrome, all experiments
                if i == 0 and n == 0:
                    fig3b, ax3b = setup_plot(n, i)
                plot_cells(n, i, fig3b, ax3b, cell_colors=dataset_c, neuronal_mask=neuronal_mask,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                           saving_name=f"neurons_{ori}_mouse.png", **all_plots_cells_params)

                # Fifth coronal plot: All cells, monochrome, current experiment
                if n == 0:
                    fig3a, ax3a = setup_plot(n, None)
                plot_cells(n, None, fig3a, ax3a, cell_colors=dataset_c, neuronal_mask=None,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                           saving_name=f"all_{ori}_mouse_{dataset_n}.png", **plot_cells_params)

                # Seventh coronal plot: All cells, monochrome, all experiments
                if i == 0 and n == 0:
                    fig3ab, ax3ab = setup_plot(n, i)
                plot_cells(n, i, fig3ab, ax3ab, cell_colors=dataset_c, neuronal_mask=None,
                           xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                           saving_name=f"all_{ori}_mouse.png", **all_plots_cells_params)

            ############################################################################################################
            # Horizontal
            ############################################################################################################

            ori = "horizontal"
            orix, oriy = 2, 0
            xlim, ylim = 369, 512

            # Second horizontal plot: Only neurons, class colors, current experiment
            if n == 0:
                fig1c, ax1c = setup_plot(n, None)
            plot_cells(n, None, fig1c, ax1c, cell_colors=np.array(cc), neuronal_mask=neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_name=f"neurons_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

            # Forth horizontal plot: Only neurons, class colors, all experiments
            if i == 0 and n == 0:
                fig1bc, ax1bc = setup_plot(n, i)
            plot_cells(n, i, fig1bc, ax1bc, cell_colors=np.array(cc), neuronal_mask=neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_name=f"neurons_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

            # Sixth horizontal plot: All cells, class colors, current experiment
            if n == 0:
                fig1ac, ax1ac = setup_plot(n, None)
            plot_cells(n, None, fig1ac, ax1ac, cell_colors=np.array(cc), neuronal_mask=None,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_name=f"all_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

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

            # Second sagittal plot: Only neurons, class colors, current experiment
            if n == 0:
                fig2c, ax2c = setup_plot(n, None)
            plot_cells(n, None, fig2c, ax2c, cell_colors=np.array(cc), neuronal_mask=neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_name=f"neurons_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

            # Forth sagittal plot: Only neurons, class colors, all experiments
            if i == 0 and n == 0:
                fig2bc, ax2bc = setup_plot(n, i)
            plot_cells(n, i, fig2bc, ax2bc, cell_colors=np.array(cc), neuronal_mask=neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_name=f"neurons_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

            # Sixth sagittal plot: All cells, class colors, current experiment
            if n == 0:
                fig2ac, ax2ac = setup_plot(n, None)
            plot_cells(n, None, fig2ac, ax2ac, cell_colors=np.array(cc), neuronal_mask=None,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=orix,
                       saving_name=f"all_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

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

            # Second coronal plot: Only neurons, class colors, current experiment
            if n == 0:
                fig3c, ax3c = setup_plot(n, None)
            plot_cells(n, None, fig3c, ax3c, cell_colors=np.array(cc), neuronal_mask=neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                       saving_name=f"neurons_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

            # Forth coronal plot: Only neurons, class colors, all experiments
            if i == 0 and n == 0:
                fig3bc, ax3bc = setup_plot(n, i)
            plot_cells(n, i, fig3bc, ax3bc, cell_colors=np.array(cc), neuronal_mask=neuronal_mask,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                       saving_name=f"neurons_{ori}_mouse_{ccat}.png", **all_plots_cells_params)

            # Sixth coronal plot: All cells, class colors, current experiment
            if n == 0:
                fig3ac, ax3ac = setup_plot(n, None)
            plot_cells(n, None, fig3ac, ax3ac, cell_colors=np.array(cc), neuronal_mask=None,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                       saving_name=f"all_{ori}_mouse_{dataset_n}_{ccat}.png", **plot_cells_params)

            # Eighth coronal plot: All cells, class colors, all experiments
            if i == 0 and n == 0:
                fig3abc, ax3abc = setup_plot(n, i)
            plot_cells(n, i, fig3abc, ax3abc, cell_colors=np.array(cc), neuronal_mask=None,
                       xlim=xlim, ylim=ylim, orix=orix, oriy=oriy, orip=oriy,
                       saving_name=f"all_{ori}_mouse_{ccat}.png", **all_plots_cells_params)
