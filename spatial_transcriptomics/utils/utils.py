import json
import requests

import anndata
import pandas as pd

def fetch_manifest(url):
    return json.loads(requests.get(url).text)

def load_data(file_path):
    return anndata.read_h5ad(file_path, backed='r')

def load_csv(file_path):
    return pd.read_csv(file_path, dtype={"cell_label": str}).set_index('cell_label')
