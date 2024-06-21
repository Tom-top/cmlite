import os
import multiprocessing
import shutil
import json
import yaml

from functools import partial

import yaml
from natsort import natsorted
import numpy as np
import tifffile
from aicsimageio.readers import CziReader

import utils.utils as ut


def get_center_tile(n_columns):
    return int(np.floor(n_columns / 2)) if n_columns % 2 != 0 else None


def save_tile(params, side, r, side_indices, c, ci, pattern, channel, saving_directory, scan_name):
    if pattern == "z":
        tile_index = int(r * params['n_cols'] + c)
    elif pattern == "s":
        if r % 2 == 0:
            tile_index = int(r * params['n_cols'] + c)
        else:
            if side[1] == "left":
                tile_index = (int(r * params['n_cols'] + c) +
                              (params['n_cols'] - len(side_indices)))
            if side[1] == "right":
                tile_index = (int(r * params['n_cols'] + c) -
                              (params['n_cols'] - len(side_indices)))
    else:
        raise ut.CmliteError(f"Scanning pattern: {pattern} unrecognized!")
    metadata = params['reader'].get_image_dask_data("ZMYX", C=channel, I=side[0], M=tile_index, cores=25)
    tile = metadata.__array__()
    tifffile.imwrite(os.path.join(saving_directory, f"{scan_name}_[{str(r).zfill(2)} x {str(ci).zfill(2)}]"
                                                    f"_{channel}.tif"), tile)


def save_right_left_tiles(side, params, sample_name, kwargs, channel, file_name, saving_directory):
    print("")
    side_indices = np.arange(np.floor(params['n_cols'] / 2), params['n_cols'], 1) if side[1] == "right" else \
        np.arange(0, np.floor(params['n_cols'] / 2), 1)
    n_tiles = params['n_rows'] * len(side_indices)
    side_tile_n = 1
    center_tile_n = 1

    for r in range(params['n_rows']):
        si = side_indices[::-1] if kwargs["scanning_pattern"] == "s" and r % 2 != 0 else side_indices
        for c, ci in zip(si, side_indices):
            c, ci = int(c), int(ci)
            # print_c(f"[INFO {sample_name}] Saving tile: [{str(r).zfill(2)} x {str(c).zfill(2)}] ({side[1]})")
            ut.print_c(f"[INFO {sample_name}] Channel {channel}: {side[1]}. Saving tile: {side_tile_n}/{n_tiles}",
                       end="\r")
            side_tile_n += 1
            save_tile(params, side, r, side_indices, c, ci, kwargs["scanning_pattern"], channel,
                      saving_directory, file_name.split(".")[0])

    for r in range(params['n_rows']):
        # Add blending and other processing steps here if necessary
        if params['c_tile'] is not None:
            # print_c(f"[INFO {sample_name}] Saving tile: [{str(r).zfill(2)} x {str(params['c_tile']).zfill(2)}]"
            #         f" (temp center {side[1]})")
            ut.print_c(f"[INFO {sample_name}] Saving tile: {center_tile_n}/{params['n_rows']}"
                       f" (temp center {side[1]})", end="\r")
            center_tile_n += 1
            center_tile_index = int(r * params['n_cols'] + params['c_tile'])
            metadata = params['reader'].get_image_dask_data("ZMYX", C=channel, I=side[0], M=center_tile_index, cores=25)
            center_tile = metadata.__array__()
            scan_name = params['scan_name'].split(".")[0]
            tifffile.imwrite(os.path.join(params["temp_directory"], f"{scan_name}_[{str(r).zfill(2)} x "
                                                                    f"{str(params['c_tile']).zfill(2)}]_{channel}_{side[1]}.tif"),
                             center_tile)


def blend_center_tile(f, params, saving_directory):
    tile_name = "_".join(f.split("_")[:-1])
    tile_idx = f.split("_")[-3]
    ut.print_c(f"[INFO] Blending left/right tiles: {tile_idx}")
    right_f_path = os.path.join(params["temp_directory"], f)
    right_data = tifffile.imread(right_f_path)
    left_f_name = "_".join([tile_name, "left.tif"])
    left_f_path = os.path.join(params["temp_directory"], left_f_name)
    left_data = tifffile.imread(left_f_path)
    blend_data = np.max([right_data, left_data], axis=0).astype("uint16")
    tifffile.imwrite(os.path.join(saving_directory, f"{tile_name}.tif"), blend_data)


def save_scan_metadata(metadata, output_directory, filename="scan_metadata.json"):
    """
    Save metadata to a JSON file.

    :param metadata: Dictionary containing metadata.
    :param output_directory: Directory where the JSON file will be saved.
    :param filename: Name of the JSON file.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_path = os.path.join(output_directory, filename)

    with open(file_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)


def prepare_samples(raw_directory, **kwargs):
    if kwargs["re_process"]:
        if len(os.listdir(raw_directory)) == 0:
            ut.CmliteError(f"No samples were found in: {raw_directory}")

        sample_names = natsorted(kwargs["samples_to_process"]) if kwargs["samples_to_process"] else os.listdir(
            raw_directory)
        for sample_name in sample_names:
            sample_directory = os.path.join(raw_directory, sample_name)
            prepare_sample(raw_directory, sample_name, **kwargs)
            if os.path.exists(os.path.join(sample_directory, "temp")):
                os.rmdir(os.path.join(sample_directory, "temp"))


def prepare_sample(raw_directory, sample_name, **kwargs):
    print("")
    sample_directory = os.path.join(raw_directory, sample_name)

    if kwargs["scanning_system"] == "zeiss":
        ut.print_c(f"[INFO {sample_name}] Starting fetching tiles!")
        file_names = [x for x in os.listdir(sample_directory) if x.endswith(".czi")]
        if not file_names:
            ut.print_c(f"[WARNING {sample_name}] No .czi file was found for sample {sample_name}: skipping!")
            return

        file_name = file_names[0]
        data_path = os.path.join(sample_directory, file_name)
        reader = CziReader(data_path)

        scan_metadata = {
            "tile_x": reader.dims.X,
            "tile_y": reader.dims.Y,
            "x_res": reader.physical_pixel_sizes.X,
            "y_res": reader.physical_pixel_sizes.Y,
            "z_res": reader.physical_pixel_sizes.Z,
        }

        for elem in reader.metadata.iter():
            if 'TileAcquisitionOverlap' in elem.tag:
                scan_metadata["overlap"] = float(elem.text)

        with open(os.path.join(sample_directory, "scan_metadata.json"), 'w') as json_file:
            json.dump(scan_metadata, json_file, indent=4)

        try:
            n_tiles = reader.dims["M"][0]
        except RuntimeError:
            ut.print_c(f"[WARNING {sample_name}] The scan for sample {sample_name} is invalid. Perhaps metadata is corrupt."
                       f" Skipping this sample")
            return

        columns = int(np.sqrt(n_tiles))
        rows = n_tiles // columns
        ut.print_c(f"[INFO {sample_name}] N tiles: {n_tiles}; rows:{rows}; columns:{columns}")

        params = {
            "sample_directory": sample_directory,
            "reader": reader,
            "n_rows": rows,
            "n_cols": columns,
            "sides": [[0, "left"], [1, "right"]],
        }

        params["c_tile"] = get_center_tile(params["n_cols"])
        params["temp_directory"] = ut.create_dir(os.path.join(params["sample_directory"], "temp"), verbose=False)

        for channel in kwargs["channels_to_stitch"]:
            saving_directory = os.path.join(sample_directory, f"processed_tiles_{channel}")
            if not os.path.exists(saving_directory):
                ut.create_dir(saving_directory, verbose=False)

                import time
                partial_func = partial(save_right_left_tiles, params=params, sample_name=sample_name, kwargs=kwargs,
                                       channel=channel, file_name=file_name, saving_directory=saving_directory)
                with multiprocessing.Pool(1) as pool:
                    pool.map(partial_func, params["sides"])

                if params["c_tile"] is not None:
                    file_names = []
                    ut.print_c(f"[INFO {sample_name}] Blending middle tiles from scan: {params['scan_name']}")
                    for f in os.listdir(params["temp_directory"]):
                        if f.split("_")[-1] == "right.tif":
                            file_names.append(f)

                    partial_blend_tile = partial(blend_center_tile, params=params, saving_directory=saving_directory)
                    with multiprocessing.Pool(1) as pool:
                        pool.map(partial_blend_tile, file_names)

                    shutil.rmtree(params["temp_directory"])

            else:
                ut.print_c(
                    f"[WARNING {sample_name}] Skipping tile fetching for channel {channel}: processed_tiles_{channel}"
                    f" folder already exists!")

    elif kwargs["scanning_system"] == "3i":
        image_record = os.path.join(sample_directory, "ImageRecord.yaml")
        with open(image_record, 'r') as yaml_file:
            yaml_data = yaml_file.read()
        yaml_documents = yaml_data.strip().split('---')[1].split('StartClass:')
        # Initialize a list to store parsed data
        # Process each YAML document individually
        for doc in yaml_documents:
            doc = "StartClass:" + doc
            # Load the document as YAML
            data = yaml.safe_load(doc)
            # Append to the parsed_data list if valid data is loaded
            try:
                pixel_size = data["StartClass"]["mMicronPerPixel"]
            except:
                scan_metadata = {}
        scan_metadata = {
            "x_res": 6,
            "y_res": 2.995,
            "z_res": 2.995,
        }
        with open(os.path.join(sample_directory, "scan_metadata.json"), 'w') as json_file:
            json.dump(scan_metadata, json_file, indent=4)
