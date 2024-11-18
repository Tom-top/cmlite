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


def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Attempt to load the JSON content
            return json.load(file)
    except FileNotFoundError:
        print("Error: The file was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def find_dict_by_key_value(data, target_id, key="id"):
    # If data is a dictionary, check if it contains the key with the matching target value
    if isinstance(data, dict):
        if key in data and data[key] == target_id:
            # Return the entire dictionary if key with matching value is found
            return data
        else:
            # Otherwise, search recursively in values
            for v in data.values():
                result = find_dict_by_key_value(v, target_id, key=key)
                if result is not None:
                    return result

    # If data is a list, iterate and search recursively in each item
    elif isinstance(data, list):
        for item in data:
            result = find_dict_by_key_value(item, target_id, key=key)
            if result is not None:
                return result

    # Return None if the key with the matching value is not found in any dictionary
    return None


def gather_ids(data):
    ids = [data['id']]
    for child in data.get('children', []):
        ids.extend(gather_ids(child))
    return ids


def find_child_ids(data, id):
    # Find the node with the given id
    id_data = find_dict_by_key_value(data, id, key="id")

    # Gather all ids starting from the found node
    if id_data and id_data['id'] == id:
        return gather_ids(id_data)[1:]  # [1:] to exclude the id of the current node
    else:
        return None
