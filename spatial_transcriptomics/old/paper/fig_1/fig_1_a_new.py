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

import utils.utils as ut

from spatial_transcriptomics.old.utils.coordinate_manipulation import filter_points_in_3d_mask
import spatial_transcriptomics.old.utils.plotting as st_plt

ATLAS_USED = "gubra"
# DATASETS = np.arange(1, 6, 1)
DATASETS = np.array([5])
N_DATASETS = len(DATASETS)
DATASET_COLORS = ["#00FF2E", "#FF00D1", "#000FFF", "#FFF000", "black"]
CATEGORY_NAMES = ["neurotransmitter", "class", "subclass", "supertype", "cluster"]
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]
BILATERAL = False  # If True: generate bilateral cell distribution in the 3D representations
ONLY_NEURONS = True  # If True: only generate plots for neurons, excluding all non-neuronal cells
PLOT_MOST_REPRESENTED_CATEGORIES = True
PERCENTAGE_THRESHOLD = 50
categories = ["class", "subclass", "supertype", "cluster", "neurotransmitter"]

ANO_DIRECTORY = r"resources\atlas"
ANO_PATH = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.tif")
ANO = np.transpose(tifffile.imread(ANO_PATH), (1, 2, 0))
ANO_JSON = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.json")

DOWNLOAD_BASE = r"/default/path"  # PERSONAL
MAP_DIR = ut.create_dir(rf"/default/path")  # PERSONAL
WHOLE_REGION = True  # If true, the unprocessed mask will be used
PLOT_COUNTS_BY_CATEGORY = False  # If true plots the category plot

labels = [1]
if WHOLE_REGION:
    TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"hemisphere_mask.tif"))
else:
    TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"smoothed_mask.tif"))
TISSUE_MASKS = [TISSUE_MASK]

MAIN_RESULTS_DIR = os.path.join(MAP_DIR, "results_4")
RESULTS_DIR = ut.create_dir(os.path.join(MAIN_RESULTS_DIR, "3d_views"))

# TRANSFORM_DIR = r"resources/abc_atlas"
TRANSFORM_DIR = r"E:\tto\mapping_aba_to_gubra_3"
# TRANSFORM_DIR = r"resources\abc_atlas"
REFERENCE_FILE = fr"resources/atlas/{ATLAS_USED}_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)
REFERENCE_SHAPE = REFERENCE.shape

ABC_ATLAS_DIRECTORY = r"resources\abc_atlas"
CLASS_COUNTS = pd.read_excel(os.path.join(ABC_ATLAS_DIRECTORY, "counts_cells_class.xlsx"))
SUBCLASS_COUNTS = pd.read_excel(os.path.join(ABC_ATLAS_DIRECTORY, "counts_cells_subclass.xlsx"))
SUPERTYPE_COUNTS = pd.read_excel(os.path.join(ABC_ATLAS_DIRECTORY, "counts_cells_supertype.xlsx"))
CLUSTER_COUNTS = pd.read_excel(os.path.join(ABC_ATLAS_DIRECTORY, "counts_cells_cluster.xlsx"))
NEUROTRANSMITTER_COUNTS = pd.read_excel(os.path.join(ABC_ATLAS_DIRECTORY, "counts_cells_neurotransmitter.xlsx"))

########################################################################################################################
# ITERATE OVER EVERY BLOB (LABEL)
########################################################################################################################

for ul, TISSUE_MASK in zip(labels, TISSUE_MASKS):

    region_id_counts = Counter(ANO[TISSUE_MASK == 255])

    most_common_region_id = max(region_id_counts, key=region_id_counts.get)
    structure = ut.read_ano_json(ANO_JSON)
    region_acronym = ut.find_key_by_id(structure, most_common_region_id, key="acronym")

    with open(ANO_JSON, 'r') as f:
        json_data = json.load(f)
    structure = json_data['msg'][0]

    sorted_region_id_counts = dict(sorted(region_id_counts.items(), key=lambda item: item[1], reverse=True))
    ids = list(sorted_region_id_counts.keys())

    # Divide the values by 2 as we are working on hemispheres
    region_id_counts_total = [int(np.sum(ANO == id)/2) for id in ids]
    acros = [ut.find_key_by_id(structure, id, key="acronym") for id in ids]
    colors = [ut.hex_to_rgb(ut.find_key_by_id(structure, id, key="color_hex_triplet")) for id in ids]

    ####################################################################################################################
    # ITERATE OVER EVERY DATASET
    ####################################################################################################################

    all_animals_data = []
    all_non_neurons = []

    for i, (dataset_n, dataset_c) in enumerate(zip(DATASETS, DATASET_COLORS)):

        ut.print_c(f"[INFO] Loading dataset: {dataset_n}")

        # Select the correct dataset
        if dataset_n < 5:
            dataset_id = f"Zhuang-ABCA-{dataset_n}"
        else:
            dataset_id = f"MERFISH-C57BL6J-638850"
        url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
        manifest = json.loads(requests.get(url).text)
        metadata = manifest['file_listing'][dataset_id]['metadata']

        # Fetch labels for each cell in the selected dataset
        metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']
        cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
        cell_metadata_file_ccf = os.path.join(DOWNLOAD_BASE, cell_metadata_path_ccf)
        cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
        cell_metadata_ccf.set_index('cell_label', inplace=True)
        cell_labels = cell_metadata_ccf.index

        # Fetch metadata ofr each for each cell in the selected dataset (class, subclass, supertype, cluster...)
        metadata_with_clusters = metadata['cell_metadata_with_cluster_annotation']
        cell_metadata_path_views = metadata_with_clusters['files']['csv']['relative_path']
        cell_metadata_file_views = os.path.join(DOWNLOAD_BASE, cell_metadata_path_views)
        cell_metadata_views = pd.read_csv(cell_metadata_file_views, dtype={"cell_label": str})
        cell_metadata_views.set_index('cell_label', inplace=True)

        # Fetch the transformed coordinates from the selected dataset
        transformed_coordinates = np.load(os.path.join(TRANSFORM_DIR, f"all_transformed_cells_{ATLAS_USED}_{dataset_n}.npy"))

        # Pre-calculate the chunks to run through in the selected dataset
        chunk_size = 10  # Size of each data chunk (voxels)
        chunks_start = np.arange(0, TISSUE_MASK.shape[0], chunk_size)
        chunks_end = np.arange(chunk_size, TISSUE_MASK.shape[0], chunk_size)
        if chunks_end[-1] != TISSUE_MASK.shape[0]:
            chunks_end = np.append(chunks_end, TISSUE_MASK.shape[0])
        n_chunks = len(chunks_start)  # Number of chunks

        # Create buffers for the selected dataset
        filtered_points_dataset = []
        # Neurotransmitter
        cells_neurotransmitter_dataset = []
        cells_neurotransmitter_colors_dataset = []
        # Class
        cells_class_dataset = []
        cells_class_colors_dataset = []
        # Subclass
        cells_subclass_dataset = []
        cells_subclass_colors_dataset = []
        # Supertype
        cells_supertype_dataset = []
        cells_supertype_colors_dataset = []
        # Cluster
        cells_cluster_dataset = []
        cells_cluster_colors_dataset = []

        ################################################################################################################
        # ITERATE OVER EVERY CHUNK IN THE SELECTED DATASET
        ################################################################################################################

        for n, (cs, ce) in enumerate(zip(chunks_start, chunks_end)):
            ut.print_c(f"[INFO] Processing chunk: {cs}:{ce}. {n}/{n_chunks}", end="\r")

            # Generate chunk mask
            chunk_mask = TISSUE_MASK.copy()
            chunk_mask[0:cs] = 0
            chunk_mask[ce:] = 0

            # Fetch the cell coordinates within the chunk
            filtered_points, mask_point = filter_points_in_3d_mask(transformed_coordinates, chunk_mask)
            filtered_points_dataset.extend(filtered_points)
            filtered_labels = np.array(cell_labels)[::-1][mask_point]
            filtered_metadata_views = cell_metadata_views.loc[filtered_labels]

            # Extract data for each category
            # Neurotransmitter
            cells_neurotransmitter = filtered_metadata_views["neurotransmitter"].tolist()
            cells_neurotransmitter_color = filtered_metadata_views["neurotransmitter_color"].tolist()
            cells_neurotransmitter_dataset.extend(cells_neurotransmitter)
            cells_neurotransmitter_colors_dataset.extend(cells_neurotransmitter_color)
            # Class
            cells_class = filtered_metadata_views["class"].tolist()
            cells_class_color = filtered_metadata_views["class_color"].tolist()
            cells_class_dataset.extend(cells_class)
            cells_class_colors_dataset.extend(cells_class_color)
            # Subclass
            cells_subclass = filtered_metadata_views["subclass"].tolist()
            cells_subclass_color = filtered_metadata_views["subclass_color"].tolist()
            cells_subclass_dataset.extend(cells_subclass)
            cells_subclass_colors_dataset.extend(cells_subclass_color)
            # Supertype
            cells_supertype = filtered_metadata_views["supertype"].tolist()
            cells_supertype_color = filtered_metadata_views["supertype_color"].tolist()
            cells_supertype_dataset.extend(cells_supertype)
            cells_supertype_colors_dataset.extend(cells_supertype_color)
            # Cluster
            cells_cluster = filtered_metadata_views["cluster"].tolist()
            cells_cluster_color = filtered_metadata_views["cluster_color"].tolist()
            cells_cluster_dataset.extend(cells_cluster)
            cells_cluster_colors_dataset.extend(cells_cluster_color)

        # Get unique occurrences in each category
        # Neurotransmitter
        unique_cells_neurotransmitter, unique_indices = np.unique(cells_neurotransmitter_dataset, return_index=True)
        unique_cells_neurotransmitter_color = np.array(cells_neurotransmitter_colors_dataset)[unique_indices]
        # Class
        unique_cells_class, unique_indices = np.unique(cells_class_dataset, return_index=True)
        unique_cells_class_color = np.array(cells_class_colors_dataset)[unique_indices]
        # Subclass
        unique_cells_subclass, unique_indices = np.unique(cells_subclass_dataset, return_index=True)
        unique_cells_subclass_color = np.array(cells_subclass_colors_dataset)[unique_indices]
        # Supertype
        unique_cells_supertype, unique_indices = np.unique(cells_supertype_dataset, return_index=True)
        unique_cells_supertype_color = np.array(cells_supertype_colors_dataset)[unique_indices]
        # Cluster
        unique_cells_cluster, unique_indices = np.unique(cells_cluster_dataset, return_index=True)
        unique_cells_cluster_color = np.array(cells_cluster_colors_dataset)[unique_indices]

        # Create masks for neuronal cells
        non_neuronal_mask_global = np.array(
            [True if any([j in i for j in NON_NEURONAL_CELL_TYPES]) else False for i in cells_class_dataset])

        ########################################################################################################
        # PLOT CELLS IN 3D
        ########################################################################################################

        filtered_points_dataset_conc = np.array([])
        filtered_points_dataset = np.array(filtered_points_dataset)
        cell_size = 0.5

        ut.print_c(f"[INFO] Cells in dataset {dataset_n}: {len(filtered_points_dataset)}")
        ut.print_c(f"[INFO] Neurons in dataset {dataset_n}: {len(filtered_points_dataset[~non_neuronal_mask_global])}")

        if BILATERAL:
            mirrored_filtered_points = filtered_points_dataset.copy()
            if mirrored_filtered_points.size > 0:
                mirrored_filtered_points[:, 2] = REFERENCE.shape[0] - 1 - filtered_points_dataset[:, 2]
            if filtered_points_dataset.shape[0] != filtered_points_dataset_conc.shape[0]:
                filtered_points_dataset_conc = np.concatenate([filtered_points_dataset, mirrored_filtered_points])
            if non_neuronal_mask_global.shape[0] != filtered_points_dataset_conc.shape[0]:
                non_neuronal_mask_global = np.tile(non_neuronal_mask_global, 2)
        else:
            filtered_points_dataset_conc = filtered_points_dataset

        all_animals_data.append(filtered_points_dataset_conc)
        all_non_neurons.append(non_neuronal_mask_global)

        ########################################################################################################
        # HORIZONTAL 3D VIEW
        ########################################################################################################

        SAVING_DIR = ut.create_dir(os.path.join(RESULTS_DIR, f"mouse_{dataset_n}"))
        sg = 0.1

        ori = "horizontal"
        orix, oriy, mask_axis = 2, 0, 1
        xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[1]

        if not ONLY_NEURONS:
            # All cells, class colors, all experiments
            st_plt.plot_cells(filtered_points_dataset_conc, REFERENCE, TISSUE_MASK, cell_colors=dataset_c,
                              cell_categories=None, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                              orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                              sg=sg, saving_path=os.path.join(SAVING_DIR, f"all_{ori}_mouse.png"),
                              relevant_categories=[], show_outline=False, zoom=False,
                              plot_individual_categories=False)
        # Only neurons, class colors, all experiments
        st_plt.plot_cells(filtered_points_dataset_conc, REFERENCE, TISSUE_MASK, cell_colors=dataset_c,
                          cell_categories=None, non_neuronal_mask=non_neuronal_mask_global, xlim=xlim,
                          ylim=ylim,
                          orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                          sg=sg, saving_path=os.path.join(SAVING_DIR, f"neurons_{ori}_mouse.png"),
                          relevant_categories=[], show_outline=False, zoom=False,
                          plot_individual_categories=False)

        ########################################################################################################
        # SAGITTAL 3D VIEW
        ########################################################################################################

        ori = "sagittal"
        orix, oriy, mask_axis = 0, 1, 2
        xlim, ylim = REFERENCE_SHAPE[1], REFERENCE_SHAPE[2]

        if not ONLY_NEURONS:
            # All cells, class colors, all experiments
            st_plt.plot_cells(filtered_points_dataset_conc, REFERENCE, TISSUE_MASK, cell_colors=dataset_c,
                              cell_categories=None, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                              orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                              sg=sg, saving_path=os.path.join(SAVING_DIR, f"all_{ori}_mouse.png"),
                              relevant_categories=[], show_outline=False, zoom=False,
                              plot_individual_categories=False)
        # Only neurons, class colors, all experiments
        st_plt.plot_cells(filtered_points_dataset_conc, REFERENCE, TISSUE_MASK, cell_colors=dataset_c,
                          cell_categories=None, non_neuronal_mask=non_neuronal_mask_global, xlim=xlim, ylim=ylim,
                          orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                          sg=sg, saving_path=os.path.join(SAVING_DIR, f"neurons_{ori}_mouse.png"),
                          relevant_categories=[], show_outline=False, zoom=False,
                          plot_individual_categories=False)


        ########################################################################################################
        # CORONAL 3D VIEW
        ########################################################################################################

        ori = "coronal"
        orix, oriy, mask_axis = 2, 1, 0  # Projection = 1
        xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[2]

        if not ONLY_NEURONS:
            # All cells, class colors, all experiments
            st_plt.plot_cells(filtered_points_dataset_conc, REFERENCE, TISSUE_MASK, cell_colors=dataset_c,
                              cell_categories=None, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                              orix=orix, oriy=oriy, orip=oriy, ori=ori, mask_axis=mask_axis, s=cell_size,
                              sg=sg, saving_path=os.path.join(SAVING_DIR, f"all_{ori}_mouse.png"),
                              relevant_categories=[], show_outline=False, zoom=False,
                              plot_individual_categories=False)
        # Only neurons, class colors, all experiments
        st_plt.plot_cells(filtered_points_dataset_conc, REFERENCE, TISSUE_MASK, cell_colors=dataset_c,
                          cell_categories=None, non_neuronal_mask=non_neuronal_mask_global, xlim=xlim,
                          ylim=ylim,
                          orix=orix, oriy=oriy, orip=oriy, ori=ori, mask_axis=mask_axis, s=cell_size,
                          sg=sg, saving_path=os.path.join(SAVING_DIR, f"neurons_{ori}_mouse.png"),
                          relevant_categories=[], show_outline=False, zoom=False,
                          plot_individual_categories=False)

    all_colors = []
    for animal_data, animal_color in zip(all_animals_data, DATASET_COLORS):
        all_colors.append([animal_color]*len(animal_data))
    all_animals_data = np.concatenate(all_animals_data)
    all_non_neurons = np.concatenate(all_non_neurons)
    all_colors = np.concatenate(all_colors)

    ori = "horizontal"
    orix, oriy, mask_axis = 2, 0, 1
    xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[1]

    if not ONLY_NEURONS:
        # All cells, class colors, all experiments
        st_plt.plot_cells(all_animals_data, REFERENCE, TISSUE_MASK, cell_colors=all_colors,
                          cell_categories=None, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                          orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                          sg=sg, saving_path=os.path.join(os.path.dirname(SAVING_DIR), f"all_{ori}_merge.png"),
                          relevant_categories=[], show_outline=False, zoom=False,
                          plot_individual_categories=False)
    # Only neurons, class colors, all experiments
    st_plt.plot_cells(all_animals_data, REFERENCE, TISSUE_MASK, cell_colors=all_colors,
                      cell_categories=None, non_neuronal_mask=all_non_neurons, xlim=xlim,
                      ylim=ylim,
                      orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                      sg=sg, saving_path=os.path.join(os.path.dirname(SAVING_DIR), f"neurons_{ori}_merge.png"),
                      relevant_categories=[], show_outline=False, zoom=False,
                      plot_individual_categories=False)

    ########################################################################################################
    # SAGITTAL 3D VIEW
    ########################################################################################################

    ori = "sagittal"
    orix, oriy, mask_axis = 0, 1, 2
    xlim, ylim = REFERENCE_SHAPE[1], REFERENCE_SHAPE[2]

    if not ONLY_NEURONS:
        # All cells, class colors, all experiments
        st_plt.plot_cells(all_animals_data, REFERENCE, TISSUE_MASK, cell_colors=all_colors,
                          cell_categories=None, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                          orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                          sg=sg, saving_path=os.path.join(os.path.dirname(SAVING_DIR), f"all_{ori}_merge.png"),
                          relevant_categories=[], show_outline=False, zoom=False,
                          plot_individual_categories=False)
    # Only neurons, class colors, all experiments
    st_plt.plot_cells(all_animals_data, REFERENCE, TISSUE_MASK, cell_colors=all_colors,
                      cell_categories=None, non_neuronal_mask=all_non_neurons, xlim=xlim, ylim=ylim,
                      orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                      sg=sg, saving_path=os.path.join(os.path.dirname(SAVING_DIR), f"neurons_{ori}_merge.png"),
                      relevant_categories=[], show_outline=False, zoom=False,
                      plot_individual_categories=False)

    ########################################################################################################
    # CORONAL 3D VIEW
    ########################################################################################################

    ori = "coronal"
    orix, oriy, mask_axis = 2, 1, 0  # Projection = 1
    xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[2]

    if not ONLY_NEURONS:
        # All cells, class colors, all experiments
        st_plt.plot_cells(all_animals_data, REFERENCE, TISSUE_MASK, cell_colors=all_colors,
                          cell_categories=None, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                          orix=orix, oriy=oriy, orip=oriy, ori=ori, mask_axis=mask_axis, s=cell_size,
                          sg=sg, saving_path=os.path.join(os.path.dirname(SAVING_DIR), f"all_{ori}_merge.png"),
                          relevant_categories=[], show_outline=False, zoom=False,
                          plot_individual_categories=False)
    # Only neurons, class colors, all experiments
    st_plt.plot_cells(all_animals_data, REFERENCE, TISSUE_MASK, cell_colors=all_colors,
                      cell_categories=None, non_neuronal_mask=all_non_neurons, xlim=xlim,
                      ylim=ylim,
                      orix=orix, oriy=oriy, orip=oriy, ori=ori, mask_axis=mask_axis, s=cell_size,
                      sg=sg, saving_path=os.path.join(os.path.dirname(SAVING_DIR), f"neurons_{ori}_merge.png"),
                      relevant_categories=[], show_outline=False, zoom=False,
                      plot_individual_categories=False)
