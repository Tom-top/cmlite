import os
import json
import requests

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

from concurrent.futures import ProcessPoolExecutor

from spatial_transcriptomics.to_check.analysis import sc_helper_functions as sc_utils

genes_to_plot = ["Fos", "Npas4", "Nr4a1", "Arc", "Egr1", "Bdnf", "Pcsk1", "Crem", "Igf1", "Scg2", "Nptx2", "Homer1",
                "Pianp", "Serpinb2", "Ostn"]
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
                        saving_directory, saving_name="_all", mask_name="", cluster_name="", vmax=2000)

# Plot Glia
with ProcessPoolExecutor(max_workers=None) as executor:
    for n, gene_name in enumerate(genes_to_plot):
        executor.submit(sc_utils.plot_gene_expression_in_10X_data, gene_name, datasets_paths, exp,
                        saving_directory, saving_name="_neurons", mask_name="non_neurons", cluster_name="")

# Plot Neurons
with ProcessPoolExecutor(max_workers=None) as executor:
    for n, gene_name in enumerate(genes_to_plot):
        executor.submit(sc_utils.plot_gene_expression_in_10X_data, gene_name, datasets_paths, exp,
                        saving_directory, saving_name="_non_neurons", mask_name="neurons", cluster_name="")

########################################################################################################################
# PLOT SPECIFIC CLUSTER
########################################################################################################################

clusters = ["0951 STR D1 Gaba_3", "0960 STR D1 Gaba_7", "0962 STR D1 Gaba_8"]
genes_to_plot = ["Slit2"]

for cluster in clusters:
    cluster_n = cluster.split(" ")[0]
    for n, gene_name in enumerate(genes_to_plot):
        sc_utils.plot_gene_expression_in_10X_data(gene_name, datasets_paths, exp,
                                                  saving_directory, saving_name=f"_cluster_{cluster_n}", mask_name="",
                                                  cluster_name=cluster, no_bg=True, vmax=500)
