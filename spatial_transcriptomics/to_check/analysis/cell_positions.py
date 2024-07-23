import os
import pandas as pd
import numpy as np
import anndata
import json
import requests
import tifffile
import nibabel as nib
from functools import reduce

from ClearMap.Environment import *
import utils as ut

dataset_id = "Zhuang-ABCA-1"
download_base = r'/mnt/data/spatial_transcriptomics'
processed_data_path = os.path.join(download_base, "results")
positions_directory = ut.create_directory(os.path.join(processed_data_path, "cell_positions"))

url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
manifest = json.loads(requests.get(url).text)

metadata = manifest['file_listing'][dataset_id]['metadata']
metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']
expression_matrices = manifest['file_listing'][dataset_id]['expression_matrices']

cell_metadata_path = expression_matrices[dataset_id]['log2']['files']['h5ad']['relative_path']
file = os.path.join(download_base, cell_metadata_path)

adata = anndata.read_h5ad(file, backed='r')
genes = adata.var

exp_thresh = 3
genes_to_visualize = [
                    # [["Lepr", np.log2(exp_thresh)], ["Slc32a1", np.log2(exp_thresh)]],
                    #   [["Lepr", np.log2(exp_thresh)], ["Slc17a7", np.log2(exp_thresh)]],
                    #   [["Slc17a7", np.log2(exp_thresh)]],
                    #   [["Lepr", np.log2(exp_thresh)]],
                    #   [["Lepr", np.log2(exp_thresh)], ["Slc17a6", np.log2(exp_thresh)]],
                    #   [["Lepr", np.log2(exp_thresh)], ["Pomc", np.log2(exp_thresh)]],
                    #   [["Lepr", np.log2(exp_thresh)], ["Agrp", np.log2(exp_thresh)]],
                      # [["Slc17a6", np.log2(exp_thresh)]],
                      #[["Slc32a1", np.log2(exp_thresh)]],
                      #[["Slc17a6", np.log2(exp_thresh)], ["Calb1", np.log2(exp_thresh)]],
                      [["Slc32a1", np.log2(exp_thresh)], ["Calb1", np.log2(exp_thresh)]],
                      ]

color_atlas_path = r"U:\Users\TTO\spatial_transcriptomics\atlas_ressources\annotation_25_full_color.tif"
color_atlas = tifffile.imread(color_atlas_path)
color_atlas = np.swapaxes(color_atlas, 1, 3)
color_atlas = np.swapaxes(color_atlas, 0, 2)
color_atlas = np.flip(color_atlas, 1)
# tifffile.imwrite(os.path.join(saving_directory, f"test.tif"), color_atlas)

# Loop over each gene
for g in genes_to_visualize:

    gene_list = [i[0] for i in g]
    genes_to_visualize_name = "_".join(gene_list)
    thresholds = [i[1] for i in g]

    # Creates saving directory for each gene
    saving_directory = ut.create_directory(os.path.join(positions_directory, genes_to_visualize_name))

    # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
    cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
    cell_metadata_file_ccf = os.path.join(download_base, cell_metadata_path_ccf)
    cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
    cell_metadata_ccf.set_index('cell_label', inplace=True)
    n_cells_ccf = len(cell_metadata_ccf)

    ####################################################################################################################
    # Loads expression data for the selected gene
    ####################################################################################################################

    gene_data_ccf = None

    for n, gene in enumerate(gene_list):
        gene_mask = [x == gene for x in genes.gene_symbol]  # Create mask for selected gene

        gene_filtered = genes[gene_mask]  # Get selected gene metadata
        gdata = adata[:, gene_filtered.index].to_df()

        gdata.columns = gene_filtered.gene_symbol  # Change columns from index to gene symbol
        pred = pd.notna(gdata[gdata.columns[0]])  # Create mask of na values in the gene expression data
        gdata = gdata[pred].copy(deep=True)  # Filter out na values from the gene expression data

        if n == 0:
            gene_data_ccf = gdata.copy()
        else:
            gene_data_ccf = gene_data_ccf.join(gdata, how='inner')

    gene_data_ccf = gene_data_ccf.join(cell_metadata_ccf, how='inner')

    ####################################################################################################################
    # Creates a 3D expression map for the selected gene in the Allen Brain Atlas (CCF) space
    ####################################################################################################################

    # Save the gene expression data
    # file = os.path.join(saving_directory, f'{g}_all_cells_expression_ccf.csv')
    # gene_data_ccf.to_csv(file)

    all_exp_data_gene_thresh = []
    for n, (gene, thresh) in enumerate(zip(gene_list, thresholds)):
        exp_data_gene_thresh = np.array(gene_data_ccf[gene]) > thresh
        print(np.array(gene_data_ccf[gene]).min(), np.array(gene_data_ccf[gene]).max())
        all_exp_data_gene_thresh.append(exp_data_gene_thresh)
    all_exp_data_gene_thresh = reduce(np.logical_and, all_exp_data_gene_thresh)

    cell_filtered_ccf = gene_data_ccf[all_exp_data_gene_thresh].reset_index()
    n_cells_filt = len(cell_filtered_ccf)
    print(f"Original n cells: {n_cells_ccf}; Filtered n cells: {n_cells_filt}")

    res = [25]  # Fixme: add 10 and 25 micron

    for r in res:
        if r == 25:
            scaling = 25 * 1.6
            shape = (528, 320, 456)
        elif r == 10:
            scaling = 25 * 4
            shape = (1320, 800, 1140)

        coordinates = []
        colors = []

        for n in range(n_cells_filt):
            cell_data = cell_filtered_ccf.loc[n]
            x, y, z = int(cell_data["x"] * scaling), int(cell_data["y"] * scaling), int(cell_data["z"] * scaling)
            if r == 10:
                color = color_atlas[int(x/2.5), int(y/2.5), int(z/2.5)]  # Fixme: Warning! This read color from atlas
            else:
                color = color_atlas[x, y, z]  # Fixme: Warning! This read color from atlas
            coordinates.append([x, y, z])
            colors.append(color)
            print(f"Fetching cell: {n + 1}/{n_cells_filt}: x:{x}, y:{y}, z:{z}")

        coordinates = np.array(coordinates)
        colors = np.array(colors)

        ####################################################################################################################
        # Transforms cells to Gubra space
        ####################################################################################################################

        transform_parameter_file = r"U:\Users\TTO\spatial_transcriptomics\atlas_ressources\gubra_to_aba\TransformParameters.1.txt"
        # coordinates = np.load(os.path.join(saving_directory, f"{g}_coordinates_ccfv3.npy"))
        coordinates = np.flip(coordinates)
        colors = colors[::-1]
        # padded_coordinates = coordinates.copy()
        # padded_coordinates[:, :, 2] = padded_coordinates[:, :, 2] + 70

        transformed_coordinates = elx.transform_points(coordinates, sink=None, indices=False,
                                                       transform_parameter_file=transform_parameter_file,
                                                       binary=False)

        np.save(os.path.join(saving_directory, f"coordinates_gubra_colors.npy"), colors)
        np.save(os.path.join(saving_directory, f"coordinates_gubra.npy"), transformed_coordinates)

        if r == 25:
            gubra_atlas_shape = np.array([369, 268, 512])
        elif r == 10:
            gubra_atlas_shape = np.array([369*2.5, 268*2.5, 512*2.5])
            gubra_atlas_shape = gubra_atlas_shape.astype(int)
        dummy = np.full(gubra_atlas_shape, 0).astype("uint8")  # Fixme: Warning! Buffer for RGB image
        dummy = np.stack([dummy] * 3, axis=-1)  # Fixme: Warning! Buffer for RGB image
        for i in range(transformed_coordinates.shape[0]):
            x, y, z = transformed_coordinates[i]
            x, y, z = int(x), int(y), int(z)
            try:
                dummy[x, y, z] = colors[i]
            except IndexError:
                print(f"Skipped pt: {transformed_coordinates[i]}")

        dummy = np.swapaxes(dummy, 0, 2)
        half = int(gubra_atlas_shape[0]/2)
        left_side = dummy[:, :, :half]
        left_side_flipped = np.flip(left_side, 2)
        if gubra_atlas_shape[0] % 2 == 0:
            dummy[:, :, half:] = left_side_flipped
        else:
            dummy[:, :, half + 1:] = left_side_flipped
        tifffile.imwrite(os.path.join(saving_directory, f"{genes_to_visualize_name}_log2{exp_thresh}_gubra_{r}.tif"), dummy)
