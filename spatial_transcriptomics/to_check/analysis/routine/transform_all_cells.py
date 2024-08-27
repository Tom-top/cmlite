import os
import json
import requests

import numpy as np
import pandas as pd
import anndata
# import tifffile

import alignment.align as elx

# dataset_ns = np.arange(1, 6, 1)
dataset_ns = np.array([2])

for n in dataset_ns:

    dataset_n = n
    if dataset_n < 5:
        dataset_id = f"Zhuang-ABCA-{dataset_n}"
    else:
        dataset_id = f"MERFISH-C57BL6J-638850"
    download_base = r'E:\tto\spatial_transcriptomics'  # PERSONAL
    url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
    manifest = json.loads(requests.get(url).text)
    metadata = manifest['file_listing'][dataset_id]['metadata']
    metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']
    expression_matrices = manifest['file_listing'][dataset_id]['expression_matrices']
    if dataset_n < 5:
        cell_metadata_path = expression_matrices[dataset_id]['log2']['files']['h5ad']['relative_path']
    else:
        cell_metadata_path = expression_matrices["-".join(dataset_id.split("-")[1:])]['log2']['files']['h5ad']['relative_path']
    file = os.path.join(download_base, cell_metadata_path)
    adata = anndata.read_h5ad(file, backed='r')
    genes = adata.var

    # color_atlas_path = r"/mnt/data/spatial_transcriptomics/atlas_ressources/ABA_25um_annotation_rgb_horizontal.tif"
    # color_atlas = tifffile.imread(color_atlas_path)
    # color_atlas = np.swapaxes(color_atlas, 1, 3)
    # color_atlas = np.swapaxes(color_atlas, 0, 2)
    # color_atlas = np.flip(color_atlas, 1)

    # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
    cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
    cell_metadata_file_ccf = os.path.join(download_base, cell_metadata_path_ccf)
    cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
    cell_metadata_ccf.set_index('cell_label', inplace=True)
    cell_labels = cell_metadata_ccf.index
    n_cells_ccf = len(cell_metadata_ccf)
    scaling = 25 * 1.6

    coordinates = []
    colors = []
    for n, cl in enumerate(cell_labels):
        cell_data = cell_metadata_ccf.loc[cl]
        x, y, z = int(cell_data["x"] * scaling), int(cell_data["y"] * scaling), int(cell_data["z"] * scaling)
        coordinates.append([x, y, z])
        # color = color_atlas[x, y, z]
        # colors.append(color)
        print(f"Fetching cell: {n + 1}/{n_cells_ccf}: x:{x}, y:{y}, z:{z}")
    coordinates = np.array(coordinates)
    colors = np.array(colors)
    new_coordinates = np.flip(coordinates)

    # transformed_coordinates = elx.transform_points_with_transformix(new_coordinates, sink=None, indices=False,
    #                                                                 transform_parameter_file=transform_parameter_file,
    #                                                                 binary=False)
    ####################################################################################################################
    # TRANSFORM THE POINTS
    ####################################################################################################################

    elx.write_points(os.path.join(elx.elastix_output_folder, "outputpoints.txt"), new_coordinates)
    transform_directory = r"resources\atlas\atlas_transformations\gubra_to_aba"

    transformed_coordinates, _ = elx.transform_points_with_transformix(
        os.path.join(elx.elastix_output_folder, "outputpoints.txt"),
        elx.elastix_output_folder,
        os.path.join(transform_directory, "TransformParameters.0.txt"),
        transformix_input=False,
    )
    # transformed_coordinates, _ = elx.transform_points_with_transformix(
    #     os.path.join(elx.elastix_output_folder, "outputpoints.txt"),
    #     elx.elastix_output_folder,
    #     os.path.join(transform_directory, "TransformParameters.1.txt"),
    #     transformix_input=True,
    # )

    transformed_coordinates_flipped = transformed_coordinates.copy()
    transformed_coordinates_flipped[:, [0, 2]] = transformed_coordinates_flipped[:, [2, 0]]
    np.save(fr"E:\tto\transformed_data\all_transformed_cells_gubra_{dataset_n}.npy",
            transformed_coordinates_flipped)  # PERSONAL
    # np.save(fr"/default/path", colors[::-1])  # PERSONAL

    # test = np.load(r"E:\tto\results\heatmaps\general\all_transformed_cells.npy")
