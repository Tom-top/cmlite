import os

import tifffile

import utils.utils as ut

from trailmap.TrailMap.inference import *
from trailmap.TrailMap.models import *

# Load the network
weights_path = r"trailmap\TrailMap\data\model-weights\trailmap_model.hdf5"

model = get_net()
model.load_weights(weights_path)

working_directory = r"E:\tto\24-VOILE-0726\ID002_an000002_g001_Brain_M3"
data_file = os.path.join(working_directory, "clean_16bit_inpainted.tif")
input_folder = ut.create_dir(os.path.join(working_directory, "input_folder"))
if not os.path.exists(input_folder):
    data = tifffile.imread(data_file)
    n_planes = len(data)
    for n, d in enumerate(data):
        ut.print_c(f"[INFO] Saving plane {n+1}/{n_planes}")
        tifffile.imwrite(os.path.join(input_folder, f"Z{str(n).zfill(4)}.tif"), d)

output_folder = ut.create_dir(os.path.join(working_directory, "output_folder"))
# Segment the brain
segment_brain(input_folder, output_folder, model)
