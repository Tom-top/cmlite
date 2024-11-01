import os
import json
import requests

import csv
import numpy as np
import pandas as pd
import tifffile
from collections import Counter
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut

from spatial_transcriptomics.utils.coordinate_manipulation import filter_points_in_3d_mask
import spatial_transcriptomics.utils.plotting as st_plt

ATLAS_USED = "gubra"
DATASETS = np.arange(1, 6, 1)
N_DATASETS = len(DATASETS)
CATEGORY_NAMES = ["neurotransmitter", "class", "subclass", "supertype", "cluster"]
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]
BILATERAL = True  # If True: generate bilateral cell distribution in the 3D representations
ONLY_NEURONS = True  # If True: only generate plots for neurons, excluding all non-neuronal cells
PLOT_MOST_REPRESENTED_CATEGORIES = False
PERCENTAGE_THRESHOLD = 15
# categories = ["class", "subclass", "supertype", "cluster", "neurotransmitter"]
categories = ["cluster"]

ANO_DIRECTORY = r"resources\atlas"
ANO_PATH = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.tif")
ANO = np.transpose(tifffile.imread(ANO_PATH), (1, 2, 0))
ANO_JSON = os.path.join(ANO_DIRECTORY, f"{ATLAS_USED}_annotation_mouse.json")

DOWNLOAD_BASE = r"/default/path"  # PERSONAL
MAP_DIR = ut.create_dir(rf"/default/path")  # PERSONAL
WHOLE_REGION = True  # If true, the unprocessed mask will be used
LABELED_MASK = False  # If true the TISSUE_MASK is a labeled 32bit mask, not a binary.
PLOT_COUNTS_BY_CATEGORY = True  # If true plots the category plot
SHOW_OUTLINE = False
ZOOM = False
SHOW_REF = True

if WHOLE_REGION:
    LABELED_MASK = False
if LABELED_MASK:  # Each label will be processed separately.
    TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"labeled_mask.tif"))
    unique_labels = np.unique(TISSUE_MASK)
    TISSUE_MASKS = [(TISSUE_MASK == ul).astype("uint8") * 255 for ul in unique_labels if not ul == 0]
    labels = [ul for ul in unique_labels if not ul == 0]
else:
    labels = [1]
    if WHOLE_REGION:
        TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"hemisphere_mask.tif"))
        # TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"peri-pag_mask.tif"))
    else:
        TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"smoothed_mask.tif"))
    TISSUE_MASKS = [TISSUE_MASK]

MAIN_RESULTS_DIR = os.path.join(MAP_DIR, "results")
RESULTS_DIR = ut.create_dir(os.path.join(MAIN_RESULTS_DIR, "3d_views"))

TRANSFORM_DIR = r"resources/abc_atlas"
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

    if LABELED_MASK:
        SAVING_DIR = os.path.join(RESULTS_DIR, f"{region_acronym}_blob_1")
        if os.path.exists(SAVING_DIR):
            n_region_dirs = len([n for n in os.listdir(RESULTS_DIR) if n.startswith(region_acronym)])
            SAVING_DIR = os.path.join(RESULTS_DIR, f"{region_acronym}_blob_{n_region_dirs + 1}")
        SAVING_DIR = ut.create_dir(SAVING_DIR)
    else:
        SAVING_DIR = RESULTS_DIR

    sorted_region_id_counts = dict(sorted(region_id_counts.items(), key=lambda item: item[1], reverse=True))
    ids = list(sorted_region_id_counts.keys())

    # Divide the values by 2 as we are working on hemispheres
    region_id_counts_total = [int(np.sum(ANO == id) / 2) for id in ids]
    acros = [ut.find_key_by_id(structure, id, key="acronym") for id in ids]
    colors = [ut.hex_to_rgb(ut.find_key_by_id(structure, id, key="color_hex_triplet")) for id in ids]
    st_plt.stacked_bar_plot_atlas_regions(sorted_region_id_counts,
                                          np.array(region_id_counts_total),
                                          np.array(acros),
                                          np.array(colors),
                                          SAVING_DIR)

    # Create buffers for the merged datasets
    filtered_points_merged = []
    # Neurotransmitter
    cells_neurotransmitter_merged = []
    cells_neurotransmitter_colors_merged = []
    # Class
    cells_class_merged = []
    cells_class_colors_merged = []
    # Subclass
    cells_subclass_merged = []
    cells_subclass_colors_merged = []
    # Supertype
    cells_supertype_merged = []
    cells_supertype_colors_merged = []
    # Cluster
    cells_cluster_merged = []
    cells_cluster_colors_merged = []

    ####################################################################################################################
    # ITERATE OVER EVERY DATASET
    ####################################################################################################################

    for i, dataset_n in enumerate(DATASETS):

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
            cells_cls_color = filtered_metadata_views["class_color"].tolist()
            cells_class_dataset.extend(cells_class)
            cells_class_colors_dataset.extend(cells_cls_color)
            # Subclass
            cells_subclass = filtered_metadata_views["subclass"].tolist()
            cells_subcls_color = filtered_metadata_views["subclass_color"].tolist()
            cells_subclass_dataset.extend(cells_subclass)
            cells_subclass_colors_dataset.extend(cells_subcls_color)
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

        # Extend the buffers for the merged datasets
        filtered_points_merged.extend(filtered_points_dataset)
        # Neurotransmitter
        cells_neurotransmitter_merged.extend(cells_neurotransmitter_dataset)
        cells_neurotransmitter_colors_merged.extend(cells_neurotransmitter_colors_dataset)
        # Class
        cells_class_merged.extend(cells_class_dataset)
        cells_class_colors_merged.extend(cells_class_colors_dataset)
        # Subclass
        cells_subclass_merged.extend(cells_subclass_dataset)
        cells_subclass_colors_merged.extend(cells_subclass_colors_dataset)
        # Supertype
        cells_supertype_merged.extend(cells_supertype_dataset)
        cells_supertype_colors_merged.extend(cells_supertype_colors_dataset)
        # Cluster
        cells_cluster_merged.extend(cells_cluster_dataset)
        cells_cluster_colors_merged.extend(cells_cluster_colors_dataset)

    # Get unique occurrences in each category
    # Neurotransmitter
    unique_cells_neurotransmitter, unique_indices = np.unique(cells_neurotransmitter_merged, return_index=True)
    unique_cells_neurotransmitter_color = np.array(cells_neurotransmitter_colors_merged)[unique_indices]
    # Class
    unique_cells_class, unique_indices = np.unique(cells_class_merged, return_index=True)
    unique_cells_class_color = np.array(cells_class_colors_merged)[unique_indices]
    # Subclass
    unique_cells_subclass, unique_indices = np.unique(cells_subclass_merged, return_index=True)
    unique_cells_subclass_color = np.array(cells_subclass_colors_merged)[unique_indices]
    # Supertype
    unique_cells_supertype, unique_indices = np.unique(cells_supertype_merged, return_index=True)
    unique_cells_supertype_color = np.array(cells_supertype_colors_merged)[unique_indices]
    # Cluster
    unique_cells_cluster, unique_indices = np.unique(cells_cluster_merged, return_index=True)
    unique_cells_cluster_color = np.array(cells_cluster_colors_merged)[unique_indices]

    ########################################################################################################################
    # CALCULATE CENTER OF MASS FOR ALL CLUSTERS
    ########################################################################################################################

    # Convert for np array
    filtered_points_merged = np.array(filtered_points_merged)

    # List to store center of mass data for each cluster
    center_of_mass_data = []
    n_unique_cells_cluster = len(unique_cells_cluster)

    for n, (cluster_name, cluster_color) in enumerate(zip(unique_cells_cluster, unique_cells_cluster_color)):
        ut.print_c(f"[INFO] Computing center of mass for cluster: {cluster_name}; {n+1}/{n_unique_cells_cluster}!")

        # Mask to filter cells in the selected cluster
        cluster_mask = np.array([True if i == cluster_name else False for i in cells_cluster_merged])

        # Filter the cells in the selected cluster
        cells_in_cluster = filtered_points_merged[cluster_mask]

        # Calculate the center of mass
        center_of_mass = np.mean(cells_in_cluster, axis=0)

        # Append data to the list
        center_of_mass_data.append([cluster_name, *center_of_mass, cluster_color])

    # Specify the CSV file path
    csv_file_path = os.path.join(RESULTS_DIR, "centers_of_mass_for_clusters.csv")

    # Write data to CSV
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Cluster Name", "Center of Mass X", "Center of Mass Y", "Center of Mass Z", "Cluster Color"])
        # Write data rows
        writer.writerows(center_of_mass_data)

    print(f"Center of mass data saved to {csv_file_path}")


########################################################################################################################
# PLOT ALL THE CLUSTERS
########################################################################################################################

cell_size = 5
cell_size_global = 5

for n, (cat_name, x, y, z, cat_color) in enumerate(center_of_mass_data):

    center_of_mass = np.array([x, y, z])
    saving_cat_name = cat_name.replace("/", "-")
    SAVING_DIR = ut.create_dir(os.path.join(RESULTS_DIR, saving_cat_name))

    ########################################################################################################
    # HORIZONTAL 3D VIEW
    ########################################################################################################
    ori = "horizontal"
    orix, oriy, mask_axis = 2, 0, 1
    xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[1]

    # Only neurons, class colors, all experiments
    st_plt.plot_cells(np.array([center_of_mass]), REFERENCE, TISSUE_MASK, cell_colors=np.array([cat_color]),
                      cell_categories=np.array([cat_name]), non_neuronal_mask=None, xlim=xlim,
                      ylim=ylim,
                      orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                      sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"cluster_{saving_cat_name}_{ori}.png"),
                      relevant_categories=[], show_outline=SHOW_OUTLINE, zoom=ZOOM,
                      show_ref=SHOW_REF)

    ########################################################################################################
    # SAGITTAL 3D VIEW
    ########################################################################################################

    ori = "sagittal"
    orix, oriy, mask_axis = 0, 1, 2
    xlim, ylim = REFERENCE_SHAPE[1], REFERENCE_SHAPE[2]

    # Only neurons, class colors, all experiments
    st_plt.plot_cells(np.array([center_of_mass]), REFERENCE, TISSUE_MASK, cell_colors=np.array([cat_color]),
                      cell_categories=np.array([cat_name]), non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                      orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                      sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"cluster_{saving_cat_name}_{ori}.png"),
                      relevant_categories=[], show_outline=SHOW_OUTLINE, zoom=ZOOM,
                      show_ref=SHOW_REF)

    ########################################################################################################
    # CORONAL 3D VIEW
    ########################################################################################################

    ori = "coronal"
    orix, oriy, mask_axis = 2, 1, 0  # Projection = 1
    xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[2]

    # Only neurons, class colors, all experiments
    st_plt.plot_cells(np.array([center_of_mass]), REFERENCE, TISSUE_MASK, cell_colors=np.array([cat_color]),
                      cell_categories=np.array([cat_name]), non_neuronal_mask=None, xlim=xlim,
                      ylim=ylim,
                      orix=orix, oriy=oriy, orip=oriy, ori=ori, mask_axis=mask_axis, s=cell_size,
                      sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"cluster_{saving_cat_name}_{ori}.png"),
                      relevant_categories=[], show_outline=SHOW_OUTLINE, zoom=ZOOM,
                      show_ref=SHOW_REF)
