import gc
import os
import tifffile
from natsort import natsorted
import utils.utils as ut
from trailmap.TrailMap.inference import *
from trailmap.TrailMap.models import *

working_directory = (r"/mnt/data/Grace/projectome/1_earmark_gfp_20tiles_2z_3p25zoom_singleill/raw/sample_1")
channel_to_segment = 0

model_name = "denardo"
models = np.arange(1, 4)

chunk_dirs = natsorted(
    [os.path.join(working_directory, i) for i in os.listdir(working_directory) if i.startswith("chunk")])

for chunk_dir in chunk_dirs:
    data_file = os.path.join(chunk_dir, "chunk.tif")
    input_folder = os.path.join(chunk_dir, "input_folder")

    if not os.path.exists(input_folder):
        os.mkdir(input_folder)
        data = tifffile.imread(data_file)
        n_planes = len(data)
        for n, d in enumerate(data):
            ut.print_c(f"[INFO] Saving plane {n + 1}/{n_planes}")
            tifffile.imwrite(os.path.join(input_folder, f"Z{str(n).zfill(4)}.tif"), d)
        del data, n_planes
        gc.collect()

    for model_n in models:
        weights_path = fr"trailmap/TrailMap/data/model-weights/{model_name}_model{model_n}.hdf5"

        # Load the model
        model = get_net()
        model.load_weights(weights_path)

        output_folder = ut.create_dir(os.path.join(chunk_dir, f"{model_name}_{model_n}_output_ch{channel_to_segment}"))

        # Segment the brain
        ut.print_c(f"[INFO] Running {model_name} on data!")
        segment_brain(input_folder, output_folder, model)

        # Clean up model and other variables
        del model
        gc.collect()

    # Clean up after each chunk is processed
    del data_file, input_folder, output_folder, weights_path
    gc.collect()
