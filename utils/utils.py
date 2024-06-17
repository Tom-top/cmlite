import os

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


def create_ws(user, experiment):
    print("\n")
    print_c(f"[INFO] User: {user}. Selected experiment: {experiment}")
    working_directory = create_dir(f"/mnt/data/{user}/{experiment}")
    raw_directory = create_dir(os.path.join(working_directory, "raw"))
    analysis_directory = create_dir(os.path.join(working_directory, "analysis"))
    return working_directory, raw_directory, analysis_directory

#
#
# def prepare_sample(raw_directory, sample_name, **kwargs):
#     print("")
#     print_c(f"[INFO {sample_name}] Fetching tiles for sample: {sample_name}")
#     sample_directory = os.path.join(raw_directory, sample_name)
#     file_names = [x for x in os.listdir(sample_directory) if x.endswith(".czi")]
#     if not file_names:
#         print_c(f"[WARNING {sample_name}] No .czi file was found for sample {sample_name}: skipping!")
#         return
#
#     file_name = file_names[0]
#     data_path = os.path.join(sample_directory, file_name)
#     reader = CziReader(data_path)
#
#     scan_metadata = {
#         "tile_x": reader.dims.X,
#         "tile_y": reader.dims.Y,
#         "x_res": reader.physical_pixel_sizes.X,
#         "y_res": reader.physical_pixel_sizes.Y,
#         "z_res": reader.physical_pixel_sizes.Z,
#     }
#
#     for elem in reader.metadata.iter():
#         if 'TileAcquisitionOverlap' in elem.tag:
#             scan_metadata["overlap"] = float(elem.text)
#
#     with open(os.path.join(sample_directory, "scan_metadata.json"), 'w') as json_file:
#         json.dump(scan_metadata, json_file, indent=4)
#
#     try:
#         n_tiles = reader.dims["M"][0]
#     except RuntimeError:
#         print_c(f"[WARNING {sample_name}] The scan for sample {sample_name} is invalid. Perhaps metadata is corrupt."
#                 f" Skipping this sample")
#         return
#
#     columns = int(np.sqrt(n_tiles))
#     rows = n_tiles // columns
#     print_c(f"[INFO {sample_name}] N tiles: {n_tiles}; rows:{rows}; columns:{columns}")
#
#     params = {
#         "sample_directory": sample_directory,
#         "reader": reader,
#         "n_rows": rows,
#         "n_cols": columns,
#         "sides": [[0, "left"], [1, "right"]],
#     }
#
#     params["c_tile"] = get_center_tile(params["n_cols"])
#     params["temp_directory"] = create_dir(os.path.join(params["sample_directory"], "temp"), verbose=False)
#
#     for channel in kwargs["channels_to_stitch"]:
#         saving_directory = os.path.join(sample_directory, f"processed_tiles_{channel}")
#         if not os.path.exists(saving_directory):
#             create_dir(saving_directory, verbose=False)
#
#             partial_func = partial(save_right_left_tiles, params=params, sample_name=sample_name, kwargs=kwargs,
#                                    channel=channel, file_name=file_name, saving_directory=saving_directory)
#             with multiprocessing.Pool(1) as pool:
#                 pool.map(partial_func, params["sides"])
#
#             if params["c_tile"] is not None:
#                 file_names = []
#                 print_c(f"[INFO {sample_name}] Blending middle tiles from scan: {params['scan_name']}")
#                 for f in os.listdir(params["temp_directory"]):
#                     if f.split("_")[-1] == "right.tif":
#                         file_names.append(f)
#
#                 partial_blend_tile = partial(blend_center_tile, params=params, saving_directory=saving_directory)
#                 with multiprocessing.Pool(1) as pool:
#                     pool.map(partial_blend_tile, file_names)
#
#                 shutil.rmtree(params["temp_directory"])
#
#         else:
#             print_c(f"[WARNING {sample_name}] Skipping tile fetching for channel {channel}: processed_tiles_{channel}"
#                     f" folder already exists!")