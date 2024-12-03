import os
import json
import requests

import pandas as pd

import utils.utils as ut

download_base = r"/default/path"  # PERSONAL

url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/20230830/manifest.json'
manifest = json.loads(requests.get(url).text)
data_id_10X = "WMB-10X"  # Select the dataset
metadata_json = manifest['file_listing'][data_id_10X]['metadata']  # Fetch metadata structure
metadata_relative_path = metadata_json['cell_metadata_with_cluster_annotation']['files']['csv']['relative_path']
metadata_file = os.path.join(download_base, metadata_relative_path)  # Path to metadata file
exp = pd.read_csv(metadata_file, low_memory=False)  # Load metadata
exp.set_index('cell_label', inplace=True)  # Set cell_label as dataframe index

metadata_genes_relative_path = metadata_json['gene']['files']['csv']['relative_path']
metadata_gene_file = os.path.join(download_base, metadata_genes_relative_path)  # Path to metadata file
genes = pd.read_csv(metadata_gene_file)  # Load metadata

gene_of_interest = "Ica1"

gene_mask = genes["gene_symbol"] == gene_of_interest
gene_ids = genes["gene_identifier"][gene_mask]
ut.print_c(f"[INFO] Identifiers for gene {gene_of_interest}: {list(gene_ids)}")
