import os

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

# PATH PARAMETERS
user = "Thomas"
experiment = "test_exp_0"

# SAMPLE PARAMETERS
parameters = dict(samples_to_process=[],
                  scanning_system="zeiss",
                  scanning_pattern="z",
                  channels_to_stitch=[0, 1],
                  channels_to_segment=[1],
                  autofluorescence_channel=0,
                  atlas_to_use="gubra",
                  stitching=dict(search_params=np.array([20, 20, 5]),
                                 z_subreg_alignment=np.array([550, 650]),
                                 ),
                  cell_detection=dict(shape_detection=700,
                                      thresholds=dict(source=None,
                                                      size=(20, 500)),
                                      ),
                  voxelization=dict(radius=(7, 7, 7))
                  )

# CREATE ESSENTIAL DIRECTORIES
working_directory, raw_directory, analysis_directory = ut.create_ws(user, experiment)

# UNZIP AND GENERATE ATLAS/TEMPLATE FILES IN THE CORRECT ORIENTATION
annotation_file, reference_file = ano.prepare_annotation_files(
    annotation_file=os.path.join("resources/atlas",
                                 f"{parameters['atlas_to_use']}_annotation.tif"),
    reference_file=os.path.join("resources/atlas",
                                f"{parameters['atlas_to_use']}_reference.tif"),
    orientation=(3, -2, -1),
    verbose=True)

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

    cells.segment_cells(sample_name, sample_directory, annotation_file, save_segmented_cells=True, **parameters)

    ####################################################################################################################
    # 4.0 VOXELIZE
    ####################################################################################################################

    vox.generate_heatmap(**parameters)
