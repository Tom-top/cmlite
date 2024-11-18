# Fixme: THIS DOES NOT WORK SO FAR. PROBLEM WITH COORDINATE ASSIGNMENT

import pandas as pd
import anndata
import json
import requests
import tifffile

from ClearMap.Environment import *
import utils as ut

dataset_id = "MERFISH-C57BL6J-638850"
download_base = r'E:\tto'
processed_data_path = os.path.join(download_base, "results")
transformed_data_directory = ut.create_directory(os.path.join(processed_data_path, "transformed_cells_to_gubra"))
dataset_directory = ut.create_directory(os.path.join(transformed_data_directory, dataset_id))
heatmaps_directory = ut.create_directory(os.path.join(processed_data_path, "heatmaps"))

url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230630/manifest.json'
manifest = json.loads(requests.get(url).text)
expression_matrices = manifest['file_listing'][dataset_id]['expression_matrices']

cell_metadata_relative_path = expression_matrices["-".join(dataset_id.split("-")[1:])]['log2']['files']['h5ad'][
    'relative_path']
cell_metadata_path = os.path.join(download_base, cell_metadata_relative_path)
adata = anndata.read_h5ad(cell_metadata_path, backed='r')
gene = adata.var
gene_names = gene.gene_symbol
gene_names = ["Th"]

for g in gene_names:  # Loop over each gene

    print(f"Generating heatmap for gene: {g}!")

    saving_directory = ut.create_directory(
        os.path.join(heatmaps_directory, g))  # Creates saving directory for each gene

    ####################################################################################################################
    # Loads expression data for the selected gene
    ####################################################################################################################

    gene_mask = [x == g for x in gene.gene_symbol]  # Create mask for selected gene
    gene_filtered = gene[gene_mask]  # Get selected gene metadata
    gdata = adata[:, gene_filtered.index].to_df()
    gdata.columns = gene_filtered.gene_symbol  # Change columns from index to gene symbol
    pred = pd.notna(gdata[gdata.columns[0]])  # Create mask of na values in the gene expression data
    gdata = gdata[pred].copy(deep=True)  # Filter out na values from the gene expression data

    ####################################################################################################################
    # Creates a 3D expression map for the selected gene in the Allen Brain Atlas (CCF) space
    ####################################################################################################################

    # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
    cell_metadata_file = os.path.join(dataset_directory, "cells_coordinates.csv")
    cell_metadata_ccf = pd.read_csv(cell_metadata_file, dtype={"cell_label": str})
    cell_metadata_ccf.set_index('cell_label', inplace=True)
    n_cells_ccf = len(cell_metadata_ccf)

    gene_data_ccf = gdata.copy()
    gene_data_ccf = gene_data_ccf.join(cell_metadata_ccf, how='inner')

    exp_data_gene_non_zero = np.array(gene_data_ccf[g]) > 0.0
    # exp_data_gene_non_zero = np.array(gene_data_ccf[g]) >= 0.0  # Fixme: Warning! This keep all cells

    cell_filtered_ccf = gene_data_ccf[exp_data_gene_non_zero].reset_index()
    n_cells_filt = len(cell_filtered_ccf)
    print(f"Original n cells: {n_cells_ccf}; Filtered n cells: {n_cells_filt}")

    intensities_file_path = os.path.join(saving_directory, f"{g}_intensities_ccfv3.npy")
    # if os.path.exists(intensities_file_path):
    #     intensities = np.load(os.path.join(saving_directory, f"{g}_intensities_ccfv3.npy"))
    # else:
    intensities = cell_filtered_ccf[g]
    np.save(os.path.join(saving_directory, f"{g}_intensities_ccfv3.npy"), cell_filtered_ccf[g])

    ####################################################################################################################
    # Voxelize
    ####################################################################################################################

    intensities = np.array(intensities)
    transformed_coordinates = np.array([cell_filtered_ccf["x_gubra"],
                                        cell_filtered_ccf["y_gubra"],
                                        cell_filtered_ccf["z_gubra"]]).T

    voxelization_parameter = dict(
        shape=(369, 268, 512),
        dtype="uint16",
        weights=intensities,
        method='sphere',
        radius=np.array([5, 5, 15]),
        kernel=None,
        processes=None,
        verbose=True
    )

    hm_path = os.path.join(saving_directory, f"{g}_heatmap_3.tif")
    if os.path.exists(hm_path):
        os.remove(hm_path)
    hm = vox.voxelize(transformed_coordinates,
                      **voxelization_parameter)
    hm = np.swapaxes(hm.array, 0, 2)
    tifffile.imwrite(hm_path, hm)

    hm = np.swapaxes(hm.array, 0, 2)
    right_side = hm[:, :, 184:]
    right_side_flipped = np.flip(right_side, 2)
    hm[:, :, :184+1] = right_side_flipped
    tifffile.imwrite(hm_path, hm)
