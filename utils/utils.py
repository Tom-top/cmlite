import os

import yaml

import settings


class CmliteError(Exception):
    pass


def print_c(msg, end="\n"):
    if msg.startswith("[INFO"):
        print(f"\033[92m{msg}\033[0m", end=end)
    elif msg.startswith("[WARNING"):
        print(f"\033[93m{msg}\033[0m", end=end)
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
    working_directory = f"/mnt/data/{kwargs['general_params']['user']}/{kwargs['study_params']['study_name']}"
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


def load_config():
    with open(settings.config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config