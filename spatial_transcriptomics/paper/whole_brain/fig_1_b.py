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
n_datasets = len(datasets)

category_names = ["class", "subclass", "supertype", "cluster"]

DOWNLOAD_BASE = r"E:\tto\spatial_transcriptomics"  # PERSONAL
MAP_DIR = r"E:\tto\spatial_transcriptomics_results"  # PERSONAL
TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"whole_brain_mask.tif"))
RESULTS_DIR = ut.create_dir(os.path.join(MAP_DIR, "results"))
CAT_DIR = ut.create_dir(os.path.join(RESULTS_DIR, "categories"))

TRANSFORM_DIR = r"resources/abc_atlas"
REFERENCE_FILE = r"resources/atlas/gubra_reference_mouse.tif"

# Get unique categories
dataset_id = f"Zhuang-ABCA-1"
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
            saving_path = ut.create_dir(saving_path)

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
                transformed_coordinates = np.load(os.path.join(TRANSFORM_DIR, f"all_transformed_cells_{dataset_n}.npy"))

                chunk_size = 10
                # overlap_size = 0
                chunks_start = np.arange(0, TISSUE_MASK.shape[0], chunk_size)
                # chunks_end = np.arange(chunk_size + overlap_size, mask.shape[0], chunk_size)
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
                    filtered_labels = cell_labels[::-1][mask_point]

                    # Get the relevant rows from cell_metadata_views in one operation
                    filtered_metadata_views = cell_metadata_views.loc[filtered_labels]
                    cat_mask = filtered_metadata_views[f"{ccat}"] == cat_v

                    filtered_points = filtered_points[cat_mask]

                    ########################################################################################################################
                    # Color transformed points
                    ########################################################################################################################

                    plot_cells_params = {
                        "n_datasets": n_datasets,
                        "reference": reference,
                        "saving_dir": CAT_DIR,
                        "filtered_points": filtered_points,
                        "n_chunks": n_chunks,
                        "marker_size": 0.1,
                        "linewidth": 0.,
                    }

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
