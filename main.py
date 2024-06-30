import os

import yaml
from natsort import natsorted
import numpy as np

import utils.utils as ut

import IO.IO as io

import stitching.tile_fetching as tf
import stitching.stitching as st

import resampling.resampling as res

import alignment.align as elx
import alignment.annotation as ano

from image_processing.experts import cells

import analysis.measurements.voxelization as vox
import analysis.statistics as stats

########################################################################################################################
# LOAD PARAMETERS
########################################################################################################################

parameters = ut.load_config()

# CREATE ESSENTIAL DIRECTORIES
working_directory, raw_directory, analysis_directory = ut.create_ws(**parameters)

# UNZIP AND GENERATE ATLAS/TEMPLATE FILES IN THE CORRECT ORIENTATION
annotation_file, reference_file = ano.prepare_annotation_files(**parameters)

########################################################################################################################
# [OPTIONAL] FETCH TILES FOR STITCHING
########################################################################################################################

tf.prepare_samples(raw_directory, **parameters)

########################################################################################################################
# [OPTIONAL] STITCHING
########################################################################################################################

st.stitch_samples(raw_directory, **parameters)

########################################################################################################################
# CONVERT TO NPY
########################################################################################################################

io.convert_stitched_files(raw_directory, **parameters)

########################################################################################################################
# START PROCESSING SAMPLES ONE BY ONE
########################################################################################################################

sample_names = natsorted(parameters["samples_to_process"]
                         if parameters["samples_to_process"] else os.listdir(raw_directory))

for sample_name in sample_names:
    sample_directory = os.path.join(raw_directory, sample_name)
    (analysis_shape_detection_directory,
     analysis_data_size_directory) = ut.create_analysis_directories(analysis_directory, **parameters)

    ####################################################################################################################
    # 1.0 RESAMPLING
    ####################################################################################################################

    res.resample_files(sample_name, sample_directory, **parameters)

    ####################################################################################################################
    # 2.0 ALIGNMENTS
    ####################################################################################################################

    elx.run_alignments(sample_name, sample_directory, annotation_file, reference_file, **parameters)

    ####################################################################################################################
    # 3.0 SEGMENT
    ####################################################################################################################

    cells.segment_cells(sample_name, sample_directory, annotation_file, reference_file, analysis_data_size_directory,
                        save_segmented_cells=True, **parameters)

    ####################################################################################################################
    # 4.0 VOXELIZE
    ####################################################################################################################

    vox.generate_heatmap(sample_directory, analysis_data_size_directory, annotation_file, weighed=False, **parameters)

########################################################################################################################
# 5.0 STATISTICS
########################################################################################################################

groups = dict(group_1=["brain_1"],
              group_2=["brain_2"], )

if parameters["statistics"]["run_statistics"]:
    for group_name, group in groups.items():
        group_sample_dirs = [os.path.join(raw_directory, i) for i in os.listdir(raw_directory) if i in group]
        group_paths = [os.path.join(i, f'shape_detection_{parameters["cell_detection"]["shape_detection"]}'
                                       f'/density_counts.tif') for i in group_sample_dirs]

        group_data = stats.read_data_group(group_paths)
        group_mean = np.mean(group_data, axis=0)
        io.write(os.path.join(raw_directory, f'{group_name}_average.tif'),
                 group_mean)
