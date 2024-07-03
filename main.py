import os

import tifffile

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

parameters = ut.load_config(config_file="custom_config.yml")

# CREATE ESSENTIAL DIRECTORIES
working_directory, raw_directory, analysis_directory = ut.create_ws(**parameters)

# UNZIP AND GENERATE ATLAS/TEMPLATE FILES IN THE CORRECT ORIENTATION
annotation_file, reference_file, metadata = ano.prepare_annotation_files(**parameters)

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

sample_names = ut.get_sample_names(raw_directory, **parameters)
(analysis_shape_detection_directory,
 analysis_data_size_directory) = ut.create_analysis_directories(analysis_directory, **parameters)

for sample_name in sample_names:
    sample_directory = io.get_sample_directory(raw_directory, sample_name, **parameters)

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
    # import numpy as np
    # x_s, y_s, z_s = 150, 300, 300
    # delta = 200
    # io.generate_npy_chunk(r"E:\tto\23-GUP030-0696-bruker\raw\ID014_an002992_g003_Brain_M3_rescan1\xy5p0_z5p0\2024-05-02_044404_merged\stitched_5.npy",
    #                       np.array([[x_s*2, (x_s+delta)*2],
    #                                 [y_s*2, (y_s+delta)*2],
    #                                 [z_s*2, (z_s+delta)*2]]),
    #                       r"E:\tto\23-GUP030-0696-bruker\raw\ID014_an002992_g003_Brain_M3_rescan1\xy5p0_z5p0\2024-05-02_044404_merged\chunk_stitched_5.npy")
    cells.segment_cells(sample_name, sample_directory, annotation_file, analysis_data_size_directory,
                        data_to_segment=r"E:\tto\23-GUP030-0696-bruker\raw\ID014_an002992_g003_Brain_M3_rescan1\xy5p0_z5p0\2024-05-02_044404_merged\chunk_stitched_5.npy",
                        save_segmented_cells=True, **parameters)

    ####################################################################################################################
    # 4.0 VOXELIZE
    ####################################################################################################################

    vox.generate_heatmap(sample_name, sample_directory, analysis_data_size_directory, annotation_file, weighed=False, **parameters)

########################################################################################################################
# 5.0 STATISTICS
########################################################################################################################

stats.run_region_wise_statistics(metadata, analysis_data_size_directory, **parameters)

stats.generate_average_maps(analysis_data_size_directory, **parameters)
stats.generate_pval_maps(analysis_data_size_directory, **parameters)