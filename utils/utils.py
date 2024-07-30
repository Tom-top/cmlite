import os
import json

import yaml
import numpy as np
from natsort import natsorted

import settings


class CmliteError(Exception):
    pass


def print_c(msg, end="\n"):
    if msg.startswith("[INFO"):
        print(f"\033[92m{msg}\033[0m", end=end)
    elif msg.startswith("[WARNING"):
        print(f"\033[93m{msg}\033[0m", end=end)
    elif msg.startswith("[CRITICAL"):
        print(f"\033[91{msg}\033[0m", end=end)
    else:
        print(f"\033[37m{msg}\033[0m", end=end)


def create_dir(dir, verbose=True):
    if os.path.exists(dir):
        if os.path.isdir(dir):
            if verbose:
                print_c(f"[WARNING] {os.path.basename(dir)} folder already exists for this experiment.")
            return dir
        else:
            raise CmliteError(f"The path '{dir} exists and is not a directory.")
    else:
        try:
            os.makedirs(dir)
            if verbose:
                print_c(f"[INFO] {os.path.basename(dir)} has been created.")
            return dir
        except:
            raise CmliteError(f"'{dir}' is not a directory and its creation failed.")


def create_ws(**kwargs):
    print("\n")
    print_c(f"[INFO] User: {kwargs['general_params']['user']}. Selected experiment:"
            f" {kwargs['study_params']['study_name']}")
    working_directory = f"{kwargs['general_params']['data_directory']}" \
                        f"/{kwargs['general_params']['user']}/{kwargs['study_params']['study_name']}"
    if not os.path.exists(working_directory):
        raise CmliteError(f"The selected experiment does not exists: {working_directory}")
    raw_directory = create_dir(os.path.join(working_directory, "raw"))
    analysis_directory = create_dir(os.path.join(working_directory, "analysis"))
    return working_directory, raw_directory, analysis_directory


def create_analysis_directories(analysis_directory, **params):
    analysis_shape_detection_directory = os.path.join(analysis_directory,
                                                      f"shape_detection_{params['cell_detection']['shape_detection']}")
    if not os.path.exists(analysis_shape_detection_directory):
        os.mkdir(analysis_shape_detection_directory)
    analysis_data_size_directory = os.path.join(analysis_shape_detection_directory,
                                                f"data_size_({params['cell_detection']['thresholds']['size'][0]},"
                                                f"{params['cell_detection']['thresholds']['size'][1]})")
    if not os.path.exists(analysis_data_size_directory):
        os.mkdir(analysis_data_size_directory)
    return analysis_shape_detection_directory, analysis_data_size_directory


def load_config(config_file=None):
    if config_file != None:
        with open(config_file, "r") as stream:
            config = yaml.safe_load(stream)
    else:
        with open(settings.config_path, "r") as stream:
            config = yaml.safe_load(stream)
    return config


def get_sample_names(raw_directory, **kwargs):
    if kwargs["study_params"]["samples_to_process"]:
        sample_names = natsorted(kwargs["study_params"]["samples_to_process"])
    else:
        sample_names = [i for i in os.listdir(raw_directory) if os.path.isdir(os.path.join(raw_directory, i))]
    return sample_names


def hex_to_rgb(hex):
    if hex == None:
        return None
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i + 2], 16)/255 for i in (0, 2, 4))


def read_ano_json(ano_json):
    with open(ano_json, 'r') as f:
        json_data = json.load(f)
    structure = json_data['msg'][0]
    return structure


def find_key_by_id(structure, target_id, key):
    """
    Recursively search for the acronym of a region with a given id in a nested JSON structure.

    :param structure: The nested structure (dictionary) to search through.
    :param target_id: The id of the region to find.
    :return: The acronym of the region if found, else None.
    """

    if structure['id'] == target_id:
        return structure[key]

    # If the structure has children, search through each child
    if 'children' in structure and structure['children']:
        for child in structure['children']:
            result = find_key_by_id(child, target_id, key)
            if result:
                return result

    return None


def assign_random_colors(grayscale_image):
    # Find unique grayscale values in the image
    unique_values = np.unique(grayscale_image)

    # Create a dictionary to map each grayscale value to a random RGB color
    color_map = {}
    for value in unique_values:
        if value != 0:
            color_map[value] = np.random.randint(0, 256, size=3)

    # Create an empty image with the same dimensions as the grayscale image, but with 3 channels (RGB)
    colored_image = np.zeros((*grayscale_image.shape, 3), dtype=np.uint8)

    # Map each grayscale value to its corresponding RGB color
    for value, color in color_map.items():
        colored_image[grayscale_image == value] = color

    return colored_image


def load_json_file(file_path):
    """
    Load and parse a JSON file.

    :param file_path: The path to the JSON file.
    :return: The parsed JSON data.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def to_raw_string(s):
    return s.replace('\\', '\\\\')
