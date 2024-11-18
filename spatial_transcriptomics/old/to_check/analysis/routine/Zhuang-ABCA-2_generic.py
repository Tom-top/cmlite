import os
import pandas as pd
import numpy as np
import anndata
import json
import requests
import tifffile
import nibabel as nib

from ClearMap.Environment import *

dataset_id = "Zhuang-ABCA-2"
download_base = r'E:\tto'
processed_data_path = os.path.join(download_base, "results")
heatmaps_directory = os.path.join(processed_data_path, "heatmaps")
if not os.path.exists(heatmaps_directory):
    os.mkdir(heatmaps_directory)
data_heatmaps_directory = os.path.join(heatmaps_directory, dataset_id)
if not os.path.exists(data_heatmaps_directory):
    os.mkdir(data_heatmaps_directory)

url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
manifest = json.loads(requests.get(url).text)

metadata = manifest['file_listing'][dataset_id]['metadata']
metadata_ccf = manifest['file_listing'][f'{dataset_id}-CCF']['metadata']
expression_matrices = manifest['file_listing'][dataset_id]['expression_matrices']

cell_metadata_path = expression_matrices[dataset_id]['log2']['files']['h5ad']['relative_path']
file = os.path.join(download_base, cell_metadata_path)

adata = anndata.read_h5ad(file, backed='r')
gene = adata.var

# gene_names = ['Lepr']
gene_names = gene.gene_symbol

# color_atlas_path = r"U:\Users\TTO\spatial_transcriptomics\atlas_ressources\annotation_25_full_color.tif"
# color_atlas = tifffile.imread(color_atlas_path)
# color_atlas = np.swapaxes(color_atlas, 1, 3)
# color_atlas = np.swapaxes(color_atlas, 0, 2)
# color_atlas = np.flip(color_atlas, 1)

# Loop over each gene
for g in gene_names:

    # Creates saving directory for each gene
    saving_directory = os.path.join(data_heatmaps_directory, g)
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)

    if not os.path.exists(os.path.join(saving_directory, f"{g}_heatmap.nii.gz")):

        print(f"Loading gene {g}")

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
        # Creates a 3D expression map for the selected gene in the slice space
        ####################################################################################################################

        # # Fetch data from cells (x, y, z) in the slice space
        # cell_metadata_path = metadata['cell_metadata']['files']['csv']['relative_path']
        # cell_metadata_file = os.path.join(download_base, cell_metadata_path)
        # cell_metadata = pd.read_csv(cell_metadata_file, dtype={"cell_label": str})
        # cell_metadata.set_index('cell_label', inplace=True)
        # n_cells = len(cell_metadata)
        #
        # gene_data = gdata.copy()
        # gene_data = gene_data.join(cell_metadata, how='inner')
        #
        # well_mapped_cells = np.array(cell_metadata['low_quality_mapping']) == False
        # exp_data_gene_non_zero = np.array(gene_data[g]) >= 0.0
        # # exp_data_gene_non_zero = np.array(gene_data[g]) > 0.0
        # cells_to_keep = np.logical_and(well_mapped_cells, exp_data_gene_non_zero)
        #
        # cell_filtered = gene_data[cells_to_keep].reset_index()
        # n_cells_filt = len(cell_filtered)
        # print(f"Original n cells: {n_cells}; Filtered n cells: {n_cells_filt}")
        #
        # new_cell_filtered_z = cell_filtered["z"].copy()
        # z_planes = np.unique(new_cell_filtered_z)
        # n_z_planes = len(z_planes)
        # new_z_planes = np.arange(n_z_planes)
        #
        # for o_z, n_z in zip(z_planes, new_z_planes):
        #     z_mask = cell_filtered["z"] == o_z
        #     print(o_z, n_z, np.sum(z_mask), new_cell_filtered_z[z_mask].index[0], new_cell_filtered_z[z_mask].index[-1])
        #     new_cell_filtered_z[z_mask] = n_z
        #
        # exp_map = np.full((n_z_planes, 1100, 1100), 0).astype("float64")
        # scaling = 100
        #
        # for n in range(n_cells_filt):
        #     cell_data = cell_filtered.loc[n]
        #     x, y, z = int(cell_data["x"] * scaling), int(cell_data["y"] * scaling), int(new_cell_filtered_z.loc[n])
        #     exp_map[z, x, y] = cell_data[g]
        #     print(f"Plotting cell: {n + 1}/{n_cells_filt}: x:{x}, y:{y}, z:{z}")
        #
        # exp_map_flip = np.flip(exp_map, 0)
        # exp_map_rot = np.rot90(exp_map_flip, -1, axes=(1, 2))
        # tifffile.imwrite(os.path.join(saving_directory, f"{g}_slices.tif"), exp_map_rot)

        ####################################################################################################################
        # Creates a 3D expression map for the selected gene in the Allen Brain Atlas (CCF) space
        ####################################################################################################################

        # Fetch data from cells (x, y, z) in the Allen Brain Atlas (CCF) space
        cell_metadata_path_ccf = metadata_ccf['ccf_coordinates']['files']['csv']['relative_path']
        cell_metadata_file_ccf = os.path.join(download_base, cell_metadata_path_ccf)
        cell_metadata_ccf = pd.read_csv(cell_metadata_file_ccf, dtype={"cell_label": str})
        cell_metadata_ccf.set_index('cell_label', inplace=True)
        n_cells_ccf = len(cell_metadata_ccf)

        gene_data_ccf = gdata.copy()
        gene_data_ccf = gene_data_ccf.join(cell_metadata_ccf, how='inner')

        # Save the gene expression data
        # file = os.path.join(saving_directory, f'{g}_all_cells_expression_ccf.csv')
        # gene_data_ccf.to_csv(file)

        exp_data_gene_non_zero = np.array(gene_data_ccf[g]) > 0.0
        # exp_data_gene_non_zero = np.array(gene_data_ccf[g]) >= 0.0  # Fixme: Warning! This keep all cells

        cell_filtered_ccf = gene_data_ccf[exp_data_gene_non_zero].reset_index()
        n_cells_filt = len(cell_filtered_ccf)
        print(f"Gene {g}. Original n cells: {n_cells_ccf}; Filtered n cells: {n_cells_filt}")

        r = 25  # Fixme: add 10 and 25 micron

        if r == 25:
            scaling = 25 * 1.6
            shape = (528, 320, 456)
        elif r == 10:
            scaling = 25 * 4
            shape = (1320, 800, 1140)

        # exp_map = np.full(shape, 0).astype("float64")
        exp_map = np.full(shape, 0).astype("int16")
        # exp_map = np.full(shape, 0).astype("uint8")  # Fixme: Warning! Buffer for RGB image
        # exp_map = np.stack([exp_map] * 3, axis=-1)  # Fixme: Warning! Buffer for RGB image

        coordinates = []
        intensities = []

        for n in range(n_cells_filt):
            cell_data = cell_filtered_ccf.loc[n]
            x, y, z = int(cell_data["x"] * scaling), int(cell_data["y"] * scaling), int(cell_data["z"] * scaling)
            # exp_map[x, y, z] = cell_data[g]
            exp_map[x, y, z] = int(cell_data[g])
            # color = color_atlas[int(x/2.5), int(y/2.5), int(z/2.5)]  # Fixme: Warning! This read color from atlas
            # exp_map[x, y, z] = color  # Fixme: Warning! This applies color from atlas
            coordinates.append([x, y, z])
            intensities.append(cell_data[g])
            print(f"Gene {g}. Fetching cell: {n + 1}/{n_cells_filt}: x:{x}, y:{y}, z:{z}")

        # tifffile.imwrite(os.path.join(saving_directory, f"{g}_ccfv3_25.tif"), exp_map)

        coordinates = np.array(coordinates)
        intensities = np.array(intensities)
        # np.save(os.path.join(saving_directory, f"{g}_coordinates_ccfv3.npy"), coordinates)
        # np.save(os.path.join(saving_directory, f"{g}_intensities_ccfv3.npy"), intensities)

        ####################################################################################################################
        # Transforms cells to Gubra space
        ####################################################################################################################

        transform_parameter_file = r"U:\Users\TTO\spatial_transcriptomics\atlas_ressources\gubra_to_aba\TransformParameters.1.txt"
        # coordinates = np.load(os.path.join(saving_directory, f"{g}_coordinates_ccfv3.npy"))
        coordinates = np.flip(coordinates)
        # padded_coordinates = coordinates.copy()
        # padded_coordinates[:, :, 2] = padded_coordinates[:, :, 2] + 70

        transformed_coordinates = elx.transform_points(coordinates, sink=None, indices=False,
                                                       transform_parameter_file=transform_parameter_file,
                                                       binary=False)

        # np.save(os.path.join(saving_directory, f"{g}_coordinates_gubra.npy"), transformed_coordinates)

        # dummy = np.full((369, 268, 512), 0).astype("uint8")
        # for i in range(transformed_coordinates.shape[0]):
        #     x, y, z = transformed_coordinates[i]
        #     x, y, z = int(x), int(y), int(z)
        #     try:
        #         dummy[x, y, z] = 255
        #     except IndexError:
        #         print(f"Skipped pt: {transformed_coordinates[i]}")
        # dummy = np.swapaxes(dummy, 0, 2)
        # tifffile.imwrite(os.path.join(saving_directory, f"{g}_gubra_25.tif"), dummy)

        ####################################################################################################################
        # Voxelize
        ####################################################################################################################

        # intensities = np.load(os.path.join(saving_directory, f"{g}_intensities_ccfv3.npy"))

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

        hm_path = os.path.join(saving_directory, f"{g}_heatmap.tif")
        if os.path.exists(hm_path):
            os.remove(hm_path)
        hm = vox.voxelize(transformed_coordinates,
                          **voxelization_parameter)

        hm = np.swapaxes(hm.array, 0, 2)
        left_side = hm[:, :, :184]
        left_side_flipped = np.flip(left_side, 2)
        hm[:, :, 184+1:] = left_side_flipped
        tifffile.imwrite(hm_path, hm)

        ####################################################################################################################
        # Convert to nii
        ####################################################################################################################

        nib_hm = np.swapaxes(hm, 1, 2)
        nib_hm = np.swapaxes(nib_hm, 0, 1)
        nib_hm = np.flip(nib_hm, 2)
        nib_hm = nib.Nifti1Image(nib_hm, np.eye(4))
        nib.save(nib_hm, os.path.join(saving_directory, f"{g}_heatmap.nii.gz"))

    else:

        print(f"Passing gene {g}. Already exists")
