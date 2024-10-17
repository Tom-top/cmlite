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
PERCENTAGE_THRESHOLD = 50
categories = ["class", "subclass", "supertype", "cluster", "neurotransmitter"]
# categories = ["cluster"]

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
        # TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"hemisphere_mask.tif"))
        TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"peri-pag_mask.tif"))
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

    # Create masks for neuronal cells
    non_neuronal_mask_global = np.array(
        [True if any([j in i for j in NON_NEURONAL_CELL_TYPES]) else False for i in cells_class_merged])

    ####################################################################################################################
    # ITERATE OVER EVERY CATEGORY
    ####################################################################################################################

    if PLOT_COUNTS_BY_CATEGORY:

        neurotransmitter_count_df = pd.DataFrame({
            'Color': np.array([]),
            'Label': np.array([]),
            'Count': np.array([]),
        })

        class_count_df = pd.DataFrame({
            'Color': np.array([]),
            'Label': np.array([]),
            'Count': np.array([]),
        })

        subclass_count_df = pd.DataFrame({
            'Color': np.array([]),
            'Label': np.array([]),
            'Count': np.array([]),
        })

        supertype_count_df = pd.DataFrame({
            'Color': np.array([]),
            'Label': np.array([]),
            'Count': np.array([]),
        })

        cluster_count_df = pd.DataFrame({
            'Color': np.array([]),
            'Label': np.array([]),
            'Count': np.array([]),
        })

        # CLASS
        if ONLY_NEURONS:
            if non_neuronal_mask_global.size > 0:
                cell_colors = np.array(cells_class_colors_merged)[~non_neuronal_mask_global]
                unique_categories, unique_indices = np.unique(np.array(cells_class_merged)[~non_neuronal_mask_global],
                                                              return_index=True)
            else:
                cell_colors = np.array([])
                unique_categories, unique_indices = np.array([]), np.array([])
        else:
            cell_colors = np.array(cells_class_colors_merged)
            unique_categories, unique_indices = np.unique(np.array(cells_class_merged),
                                                          return_index=True)
        if unique_indices.size > 0:
            unique_colors = cell_colors[unique_indices]
        else:
            unique_colors = np.array([])

        # Count the occurrences of each color in labels and sorts them
        color_counts = pd.Series(np.array(cell_colors)).value_counts() \
            .reindex(unique_colors, fill_value=0)
        df = pd.DataFrame({
            'Color': unique_colors,
            'Label': unique_categories,
            'Count': color_counts
        })
        class_sorted_df = df.sort_values(by='Count', ascending=False)
        class_count_df = pd.concat([class_count_df, class_sorted_df], axis=0)

        for n, c in enumerate(class_sorted_df["Label"]):

            class_mask = np.array(cells_class_merged) == c

            ################################################################################################################
            # SUBCLASS
            ################################################################################################################

            cell_colors = np.array(cells_subclass_colors_merged)[class_mask]
            unique_categories, unique_indices = np.unique(np.array(cells_subclass_merged)[class_mask],
                                                          return_index=True)
            unique_colors = cell_colors[unique_indices]

            # Count the occurrences of each color in labels and sorts them
            color_counts = pd.Series(np.array(cell_colors)).value_counts() \
                .reindex(unique_colors, fill_value=0)
            df = pd.DataFrame({
                'Color': unique_colors,
                'Label': unique_categories,
                'Count': color_counts
            })
            subclass_sorted_df = df.sort_values(by='Count', ascending=False)
            subclass_count_df = pd.concat([subclass_count_df, subclass_sorted_df], axis=0)

            for m, sc in enumerate(subclass_sorted_df["Label"]):

                subclass_mask = np.array(cells_subclass_merged) == sc

                ############################################################################################################
                # SUPERTYPE
                ############################################################################################################

                cell_colors = np.array(cells_supertype_colors_merged)[subclass_mask]
                unique_categories, unique_indices = np.unique(np.array(cells_supertype_merged)[subclass_mask],
                                                              return_index=True)
                unique_colors = cell_colors[unique_indices]

                # Count the occurrences of each color in labels and sorts them
                color_counts = pd.Series(np.array(cell_colors)).value_counts() \
                    .reindex(unique_colors, fill_value=0)
                df = pd.DataFrame({
                    'Color': unique_colors,
                    'Label': unique_categories,
                    'Count': color_counts
                })
                supertype_sorted_df = df.sort_values(by='Count', ascending=False)
                supertype_count_df = pd.concat([supertype_count_df, supertype_sorted_df], axis=0)

                for o, st in enumerate(supertype_sorted_df["Label"]):

                    supertype_mask = np.array(cells_supertype_merged) == st

                    ########################################################################################################
                    # CLUSTER
                    ########################################################################################################

                    cell_colors = np.array(cells_cluster_colors_merged)[supertype_mask]
                    unique_categories, unique_indices = np.unique(np.array(cells_cluster_merged)[supertype_mask],
                                                                  return_index=True)
                    unique_colors = cell_colors[unique_indices]

                    # Count the occurrences of each color in labels and sorts them
                    color_counts = pd.Series(np.array(cell_colors)).value_counts() \
                        .reindex(unique_colors, fill_value=0)
                    df = pd.DataFrame({
                        'Color': unique_colors,
                        'Label': unique_categories,
                        'Count': color_counts
                    })
                    cluster_sorted_df = df.sort_values(by='Count', ascending=False)
                    cluster_count_df = pd.concat([cluster_count_df, cluster_sorted_df], axis=0)

                    for p, cl in enumerate(cluster_sorted_df["Label"]):
                        ut.print_c(f"[INFO] Processing cells from class:{sc} {n + 1}/{len(class_sorted_df['Label'])};"
                                   f" subclass:{sc} {m + 1}/{len(subclass_sorted_df['Label'])};"
                                   f" supertype:{st} {o + 1}/{len(supertype_sorted_df['Label'])};"
                                   f" cluster:{cl} {p + 1}/{len(cluster_sorted_df['Label'])}!")
                        cluster_mask = np.array(cells_cluster_merged) == cl

                        ####################################################################################################
                        # NEUROTRANSMITTER
                        ####################################################################################################

                        cell_colors = np.array(cells_neurotransmitter_colors_merged)[cluster_mask]
                        unique_categories, unique_indices = np.unique(
                            np.array(cells_neurotransmitter_merged)[cluster_mask],
                            return_index=True)
                        unique_colors = cell_colors[unique_indices]

                        # Count the occurrences of each color in labels and sorts them
                        color_counts = pd.Series(np.array(cell_colors)).value_counts() \
                            .reindex(unique_colors, fill_value=0)
                        df = pd.DataFrame({
                            'Color': unique_colors,
                            'Label': unique_categories,
                            'Label_cluster': cl,
                            'Count': color_counts
                        })
                        neurotransmitter_sorted_df = df.sort_values(by='Count', ascending=False)
                        neurotransmitter_count_df = pd.concat([neurotransmitter_count_df, neurotransmitter_sorted_df],
                                                              axis=0)

        # SAVE THE COUNTS
        neurotransmitter_count_df.to_excel(os.path.join(SAVING_DIR, "counts_cells_neurotransmitter.xlsx"), index=False)
        class_count_df.to_excel(os.path.join(SAVING_DIR, "counts_cells_class.xlsx"), index=False)
        subclass_count_df.to_excel(os.path.join(SAVING_DIR, "counts_cells_subclass.xlsx"), index=False)
        supertype_count_df.to_excel(os.path.join(SAVING_DIR, "counts_cells_supertype.xlsx"), index=False)
        cluster_count_df.to_excel(os.path.join(SAVING_DIR, "counts_cells_cluster.xlsx"), index=False)

        merged_class_count_df = pd.merge(class_count_df, CLASS_COUNTS, on='Label', suffixes=('_df', '_all'))
        merged_class_count_df['Percentage'] = (merged_class_count_df['Count_df'] / merged_class_count_df[
            'Count_all']) * 100

        merged_subclass_count_df = pd.merge(subclass_count_df, SUBCLASS_COUNTS, on='Label', suffixes=('_df', '_all'))
        merged_subclass_count_df['Percentage'] = (merged_subclass_count_df['Count_df'] / merged_subclass_count_df[
            'Count_all']) * 100

        merged_supertype_count_df = pd.merge(supertype_count_df, SUPERTYPE_COUNTS, on='Label', suffixes=('_df', '_all'))
        merged_supertype_count_df['Percentage'] = (merged_supertype_count_df['Count_df'] / merged_supertype_count_df[
            'Count_all']) * 100

        merged_cluster_count_df = pd.merge(cluster_count_df, CLUSTER_COUNTS, on='Label', suffixes=('_df', '_all'))
        merged_cluster_count_df['Percentage'] = (merged_cluster_count_df['Count_df'] / merged_cluster_count_df[
            'Count_all']) * 100

        merged_neurotransmitter_count_df = pd.merge(neurotransmitter_count_df, NEUROTRANSMITTER_COUNTS,
                                                    on='Label_cluster', suffixes=('_df', '_all'))
        merged_neurotransmitter_count_df['Percentage'] = (merged_neurotransmitter_count_df['Count_df'] /
                                                          merged_neurotransmitter_count_df[
                                                              'Count_all']) * 100

        ut.print_c(f"[INFO] Generating plots for all categories!")
        merged_data = [merged_class_count_df, merged_subclass_count_df, merged_supertype_count_df,
                       merged_cluster_count_df, merged_neurotransmitter_count_df]

        st_plt.stacked_horizontal_bar_plot(
            categories,
            merged_data,
            SAVING_DIR,
            plots_to_generate=["categories", "categories_labeled", "percentage"],  # , "percentage"
            colormap="viridis",
        )

        ################################################################################################################
        # TEST
        ################################################################################################################

        class_data = merged_data[0]
        subclass_data = merged_data[1]
        supertype_data = merged_data[2]
        cluster_data = merged_data[3]
        hierarchy = {}

        for cd in class_data.iterrows():
            print("")
            class_label = cd[-1]["Label"]
            hierarchy[class_label] = cd[-1]  # Set the class with its data
            hierarchy[class_label]["children"] = {}  # Set a dummy dictionary for potential children
            class_counts = cd[-1]["Count_df"]
            ut.print_c(f"[INFO] Class {class_label} count: {class_counts}")

            subclass_indices = []  # Set a list for the subclass indices to be removed on the next iteration
            cumulative_subclass_counts = 0  # Counter to keep track of the cumulative counts for the subclasses
            for scd in subclass_data.iterrows():  # Iterate over each subclass
                if class_counts > cumulative_subclass_counts:
                    subclass_label = scd[-1]["Label"]
                    cumulative_subclass_counts += scd[-1]["Count_df"]
                    subclass_counts = scd[-1]["Count_df"]
                    subclass_indices.append(scd[0])
                    hierarchy[class_label]["children"][subclass_label] = scd[-1]
                    hierarchy[class_label]["children"][subclass_label]["children"] = {}  # Set a dummy dictionary for potential children

                    supertype_indices = []  # Set a list for the supertype indices to be removed on the next iteration
                    cumulative_supertype_counts = 0  # Counter to keep track of the cumulative counts for the supertypes
                    for sptd in supertype_data.iterrows():  # Iterate over each supertype
                        if subclass_counts > cumulative_supertype_counts:
                            supertype_label = sptd[-1]["Label"]
                            cumulative_supertype_counts += sptd[-1]["Count_df"]
                            supertype_counts = sptd[-1]["Count_df"]
                            supertype_indices.append(sptd[0])
                            hierarchy[class_label]["children"][subclass_label]["children"][supertype_label] = sptd[-1]
                            hierarchy[class_label]["children"][subclass_label]["children"][supertype_label]["children"] = {}

                            cluster_indices = []  # Set a list for the cluster indices to be removed on the next iteration
                            cumulative_cluster_counts = 0  # Counter to keep track of the cumulative counts for the clusters
                            for cld in cluster_data.iterrows():  # Iterate over each cluster
                                if supertype_counts > cumulative_cluster_counts:
                                    cluster_label = cld[-1]["Label"]
                                    cumulative_cluster_counts += cld[-1]["Count_df"]
                                    cluster_counts = cld[-1]["Count_df"]
                                    cluster_indices.append(cld[0])
                                    ut.print_c(
                                        f"[INFO] {class_label}: {class_counts};"
                                        f" {subclass_label}: {cumulative_subclass_counts};"
                                        f" {supertype_label}: {cumulative_supertype_counts};"
                                        f" {cluster_label}: {cumulative_cluster_counts}")
                                    hierarchy[class_label]["children"][subclass_label]["children"][supertype_label]["children"][cluster_label] = cld[-1]
                            cluster_data = cluster_data.drop(cluster_indices)
                    supertype_data = supertype_data.drop(supertype_indices)
            subclass_data = subclass_data.drop(subclass_indices)

        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib import cm
        import os
        from networkx.drawing.nx_pydot import graphviz_layout


        def filter_hierarchy_by_percentage(hierarchy, min_percentage=0.5):
            """
            Filters the hierarchy to remove clusters with a 'Percentage' <= min_percentage
            and removes supertypes without any clusters after filtering.
            If a class (root node) does not have any linked cluster, it is also removed.

            Args:
            - hierarchy (dict): The original hierarchical data.
            - min_percentage (float): The minimum percentage threshold for clusters.

            Returns:
            - dict: The filtered hierarchy.
            """
            filtered_hierarchy = {}

            # Filter each root class
            for root_class, root_data in hierarchy.items():
                filtered_subclasses = {}

                # Filter each subclass
                for subclass, subclass_data in root_data["children"].items():
                    filtered_supertypes = {}

                    # Filter each supertype
                    for supertype, supertype_data in subclass_data["children"].items():
                        # Filter clusters based on the percentage
                        filtered_clusters = {
                            cluster: cluster_data for cluster, cluster_data in supertype_data["children"].items()
                            if cluster_data.get("Percentage", 0) > min_percentage
                        }

                        # Only add the supertype if it has remaining clusters
                        if filtered_clusters:
                            filtered_supertypes[supertype] = {
                                "Label": supertype_data["Label"],
                                "Percentage": supertype_data.get("Percentage", 0),
                                # Fetch directly from the hierarchy data
                                "children": filtered_clusters
                            }

                    # Only add the subclass if it has remaining supertypes
                    if filtered_supertypes:
                        filtered_subclasses[subclass] = {
                            "Label": subclass_data["Label"],
                            "Percentage": subclass_data.get("Percentage", 0),  # Fetch directly from the hierarchy data
                            "children": filtered_supertypes
                        }

                # Only add the root class if it has remaining subclasses
                if filtered_subclasses:
                    filtered_hierarchy[root_class] = {
                        "Label": root_data["Label"],
                        "Percentage": root_data.get("Percentage", 0),  # Fetch directly from the hierarchy data
                        "children": filtered_subclasses
                    }

            return filtered_hierarchy


        def plot_linear_tree(hierarchy, save_path, percentage_thresh=40):
            """
            Plots a linear dendrogram-like tree using the hierarchy dictionary with clear separation of initial nodes.

            Args:
            - hierarchy (dict): Nested dictionary representing the hierarchical structure.
            - save_path (str): Path to save the output image.
            """
            # Create a directed graph
            G = nx.DiGraph()

            # Function to recursively add nodes and edges to the graph
            def add_edges(parent, data, level):
                """
                Recursively add nodes and edges to the graph while keeping track of levels and percentages.
                """
                # Ensure the parent node is added with its 'level' and 'percentage' attributes
                G.add_node(parent, level=level, percentage=data.get('Percentage', 0))  # Correctly assign the percentage

                for child, child_data in data.get("children", {}).items():
                    # Add each child node with its 'level' attribute and 'percentage' directly from the data
                    G.add_node(child, level=level + 1,
                               percentage=child_data.get("Percentage", 0))  # Fetch directly from data
                    G.add_edge(parent, child)
                    # Recursively add the children
                    add_edges(child, child_data, level + 1)

            # Add edges starting from the root classes in the hierarchy
            for root_class, data in hierarchy.items():
                if "children" in data:
                    add_edges(root_class, data, 0)  # Root level is 0

            # Extract levels and percentages from nodes
            levels = nx.get_node_attributes(G, 'level')
            percentages = nx.get_node_attributes(G, 'percentage')

            # Ensure all nodes have the 'level' attribute
            for node in G.nodes():
                if 'level' not in G.nodes[node]:
                    raise ValueError(f"Node {node} does not have 'level' attribute.")

            # Normalize percentages globally (0-100 range)
            min_percentage = 0
            max_percentage = percentage_thresh
            norm = plt.Normalize(vmin=min_percentage, vmax=max_percentage)
            cmap = cm.viridis  # Use viridis colormap

            # Generate a hierarchy layout for a linear tree with increased separation between levels
            pos = nx.multipartite_layout(G, subset_key='level', scale=5, align='vertical')

            # Draw the graph with clear separation of root nodes
            plt.figure(figsize=(13, 10))
            ax = plt.gca()

            # Draw edges first so they appear behind the nodes
            nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', alpha=0.7,  width=1.5)

            # Draw rectangular nodes for all nodes with width adjusted by label length
            min_x = np.inf
            max_x = -np.inf
            for node in G.nodes():
                x, y = pos[node]
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x

                # Dynamically calculate width based on text length
                label = node
                width = len(label) * 0.017  # Scale the width by a factor (adjust as needed)
                height = 0.25  # Fixed height

                # Determine color based on the globally normalized percentage
                percentage = percentages.get(node, 0)  # Use the directly fetched percentage
                color = cmap(norm(percentage))

                # Draw rectangle
                ax.add_patch(Rectangle((x - width / 2, y - height / 2), width, height,
                                       edgecolor='black', facecolor=color, lw=1, zorder=1))
                # Draw label with white text
                if percentage >= max_percentage:
                    plt.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', color='black',
                             zorder=2)
                else:
                    plt.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', color='white',
                             zorder=2)

            # Save and show the plot
            plt.title("Dendrogram-Like Tree with Viridis-Colored Nodes (Based on Percentage)")
            # plt.tight_layout()
            ax.set_xlim(min_x-0.3, max_x+0.3)
            plt.savefig(save_path)
            plt.show()


        # Filter the hierarchy based on the 'Percentage' value
        min_percentage = 10
        filtered_hierarchy = filter_hierarchy_by_percentage(hierarchy, min_percentage=min_percentage)
        # Plot the dendrogram
        plot_linear_tree(filtered_hierarchy, os.path.join(SAVING_DIR, "dendrogram.png"),
                         percentage_thresh=min_percentage)
        plot_linear_tree(filtered_hierarchy, os.path.join(SAVING_DIR, "dendrogram.svg"),
                         percentage_thresh=min_percentage)

        ################################################################################################################
        # TEST
        ################################################################################################################

        # EXTRACT THE RELEVANT CATEGORIES TO DISPLAY
        relevant_categories = []
        if PLOT_MOST_REPRESENTED_CATEGORIES:
            for d in merged_data:
                for n in range(len(d)):
                    try:
                        cat_name = d.iloc[n]["Label"]
                    except KeyError:
                        cat_name = d.iloc[n]["Label_cluster"]
                    cat_percentage = d.iloc[n]["Percentage"]
                    if cat_percentage >= PERCENTAGE_THRESHOLD:
                        relevant_categories.append(cat_name)
    else:
        relevant_categories = []

    ########################################################################################################
    # PLOT CELLS IN 3D
    ########################################################################################################

    filtered_points_merged_conc = np.array([])
    filtered_points_merged = np.array(filtered_points_merged)
    cell_size = 0.5
    cell_size_global = 0.5

    if BILATERAL:
        mirrored_filtered_points = filtered_points_merged.copy()
        if mirrored_filtered_points.size > 0:
            mirrored_filtered_points[:, 2] = REFERENCE.shape[0] - 1 - filtered_points_merged[:, 2]
        if filtered_points_merged.shape[0] != filtered_points_merged_conc.shape[0]:
            filtered_points_merged_conc = np.concatenate([filtered_points_merged, mirrored_filtered_points])
        if non_neuronal_mask_global.shape[0] != filtered_points_merged_conc.shape[0]:
            non_neuronal_mask_global = np.tile(non_neuronal_mask_global, 2)

    for n, cat in enumerate(categories):

        if BILATERAL:
            points_colors = np.tile(np.array(globals()[f"cells_{cat}_colors_merged"]), 2)
            points_cats = np.tile(np.array(globals()[f"cells_{cat}_merged"]), 2)
        else:
            points_colors = np.array(globals()[f"cells_{cat}_colors_merged"])
            points_cats = np.array(globals()[f"cells_{cat}_merged"])

        ########################################################################################################
        # HORIZONTAL 3D VIEW
        ########################################################################################################
        ori = "horizontal"
        orix, oriy, mask_axis = 2, 0, 1
        xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[1]

        if not ONLY_NEURONS:
            # All cells, class colors, all experiments
            st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                              cell_categories=points_cats, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                              orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                              sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"all_{ori}_mouse_{cat}.png"),
                              relevant_categories=relevant_categories, show_outline=SHOW_OUTLINE, zoom=ZOOM,
                              show_ref=SHOW_REF)
        # Only neurons, class colors, all experiments
        st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                          cell_categories=points_cats, non_neuronal_mask=non_neuronal_mask_global, xlim=xlim,
                          ylim=ylim,
                          orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                          sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"neurons_{ori}_mouse_{cat}.png"),
                          relevant_categories=relevant_categories, show_outline=SHOW_OUTLINE, zoom=ZOOM,
                          show_ref=SHOW_REF)

        ########################################################################################################
        # SAGITTAL 3D VIEW
        ########################################################################################################

        ori = "sagittal"
        orix, oriy, mask_axis = 0, 1, 2
        xlim, ylim = REFERENCE_SHAPE[1], REFERENCE_SHAPE[2]

        if not ONLY_NEURONS:
            # All cells, class colors, all experiments
            st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                              cell_categories=points_cats, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                              orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                              sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"all_{ori}_mouse_{cat}.png"),
                              relevant_categories=relevant_categories, show_outline=SHOW_OUTLINE, zoom=ZOOM,
                              show_ref=SHOW_REF)
        # Only neurons, class colors, all experiments
        st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                          cell_categories=points_cats, non_neuronal_mask=non_neuronal_mask_global, xlim=xlim, ylim=ylim,
                          orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                          sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"neurons_{ori}_mouse_{cat}.png"),
                          relevant_categories=relevant_categories, show_outline=SHOW_OUTLINE, zoom=ZOOM,
                          show_ref=SHOW_REF)

        ########################################################################################################
        # CORONAL 3D VIEW
        ########################################################################################################

        ori = "coronal"
        orix, oriy, mask_axis = 2, 1, 0  # Projection = 1
        xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[2]

        if not ONLY_NEURONS:
            # All cells, class colors, all experiments
            st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                              cell_categories=points_cats, non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                              orix=orix, oriy=oriy, orip=oriy, ori=ori, mask_axis=mask_axis, s=cell_size,
                              sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"all_{ori}_mouse_{cat}.png"),
                              relevant_categories=relevant_categories, show_outline=SHOW_OUTLINE, zoom=ZOOM,
                              show_ref=SHOW_REF)
        # Only neurons, class colors, all experiments
        st_plt.plot_cells(filtered_points_merged_conc, REFERENCE, TISSUE_MASK, cell_colors=points_colors,
                          cell_categories=points_cats, non_neuronal_mask=non_neuronal_mask_global, xlim=xlim,
                          ylim=ylim,
                          orix=orix, oriy=oriy, orip=oriy, ori=ori, mask_axis=mask_axis, s=cell_size,
                          sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"neurons_{ori}_mouse_{cat}.png"),
                          relevant_categories=relevant_categories, show_outline=SHOW_OUTLINE, zoom=ZOOM,
                          show_ref=SHOW_REF)
