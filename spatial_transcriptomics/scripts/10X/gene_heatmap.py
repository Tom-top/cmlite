import os
import json
import requests

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

from concurrent.futures import ProcessPoolExecutor

from spatial_transcriptomics.to_check.analysis import sc_helper_functions as sc_utils

# genes_to_plot = ["Glp1r", "Gipr", "Ramp1", "Ramp2", "Ramp3"]
# genes_to_plot = ["Slc17a6", "Klb", "Lepr", "Calcr"]
genes_to_plot = ["Slc17a6"]
# genes_to_plot = ["Hcrt", "Hcrtr1", "Hcrtr2", "Hcrtr3"]
saving_directory = r"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results" \
                   r"\10X_mapped_gene_expression"

download_base = r'E:\tto\spatial_transcriptomics'  # Path to data on the local drive
url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'  # Manifest url
manifest = json.loads(requests.get(url).text)  # Load the manifest
# Fixme: This should be fixed as only the 10Xv3 dataset is fetched (the largest). 10Xv2 and 10XMulti or omitted
dataset_id = "WMB-10Xv3"  # Dataset name
metadata_exp = manifest['file_listing'][dataset_id]['expression_matrices']

dataset_id = "WMB-10X"  # Select the dataset
metadata_json = manifest['file_listing'][dataset_id]['metadata']  # Fetch metadata structure
metadata_relative_path = metadata_json['cell_metadata_with_cluster_annotation']['files']['csv']['relative_path']
metadata_file = os.path.join(download_base, metadata_relative_path)  # Path to metadata file
exp = pd.read_csv(metadata_file)  # Load metadata
exp.set_index('cell_label', inplace=True)  # Set cell_label as dataframe index

# Pre-loading dataset paths
datasets_paths = {dsn: os.path.join(download_base, metadata_exp[dsn]["log2"]["files"]["h5ad"]["relative_path"]) for
                  dsn in metadata_exp}

# Plot all cells
with ProcessPoolExecutor(max_workers=None) as executor:
    for n, gene_name in enumerate(genes_to_plot):
        executor.submit(sc_utils.plot_gene_expression_in_10X_data, gene_name, datasets_paths, exp,
                        saving_directory, saving_name="_all", mask_name="", cluster_name="")

# Plot Glia
with ProcessPoolExecutor(max_workers=None) as executor:
    for n, gene_name in enumerate(genes_to_plot):
        executor.submit(sc_utils.plot_gene_expression_in_10X_data, gene_name, datasets_paths, exp,
                        saving_directory, saving_name="_non_neurons", mask_name="non_neurons", cluster_name="")

# Plot Neurons
with ProcessPoolExecutor(max_workers=None) as executor:
    for n, gene_name in enumerate(genes_to_plot):
        executor.submit(sc_utils.plot_gene_expression_in_10X_data, gene_name, datasets_paths, exp,
                        saving_directory, saving_name="_neurons", mask_name="neurons", cluster_name="")

########################################################################################################################
# PLOT SPECIFIC CLUSTER
########################################################################################################################

for n, gene_name in enumerate(genes_to_plot):
    sc_utils.plot_gene_expression_in_10X_data(gene_name, datasets_paths, exp,
                                              saving_directory, saving_name="_cluster_0343", mask_name="",
                                              cluster_name="0343 L2/3 IT RSP Glut_2")
