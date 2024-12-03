import os
import json
import requests

import pandas as pd
import matplotlib

matplotlib.use("Agg")

from concurrent.futures import ProcessPoolExecutor

import utils.utils as ut

from spatial_transcriptomics.old.to_check.analysis import sc_helper_functions as sc_utils

saving_directory = ut.create_dir(r"/mnt/data/Thomas/Semaglutide/10X_mapped_gene_expression")

download_base = r'/mnt/data/Thomas/data'  # Path to data on the local drive
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

sorted_and_enriched_clusters = pd.read_csv(
    "/mnt/data/Thomas/Semaglutide/results/3d_views/cluster_labels_and_percentages.csv")
clusters = [x for x, y in zip(sorted_and_enriched_clusters["Label"],
                              sorted_and_enriched_clusters["Percentage"]) if y > 40]
perc_clusters = [y for x, y in zip(sorted_and_enriched_clusters["Label"],
                              sorted_and_enriched_clusters["Percentage"]) if y > 40]
sc_utils.plot_clusters_in_10X_data(exp, saving_directory, saving_name="_Semaglutide_perc",
                                   cluster_names=clusters, no_bg=False,
                                   color=[])
