import matplotlib

matplotlib.use("Agg")  # Headless mode

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
annotation_files, reference_files, metadata_files = ano.prepare_annotation_files(**parameters)

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

    elx.run_alignments(sample_name, sample_directory, annotation_files, reference_files, **parameters)

    ####################################################################################################################
    # 3.0 SEGMENT
    ####################################################################################################################

    cells.segment_cells(sample_name, sample_directory, annotation_files, analysis_data_size_directory,
                        # data_to_segment=r"E:\tto\23-GUP030-0696\raw\ID888_an000888_g010_brain_M3\xy5p0_z5p0\2024-08-29_194534_merged\chunk_stitched_3.npy",
                        **parameters)

    ####################################################################################################################
    # 4.0 VOXELIZE
    ####################################################################################################################

    vox.generate_heatmap(sample_name, sample_directory, analysis_data_size_directory, annotation_files, weighed=False,
                         **parameters)

########################################################################################################################
# 5.0 STATISTICS
########################################################################################################################

#Fixme: Not implemented yet
#stats.run_region_wise_statistics(metadata_files, analysis_data_size_directory, **parameters)  # Fixme: metadata_files
#
#stats.generate_average_maps(analysis_data_size_directory, **parameters)
#stats.generate_pval_maps(analysis_data_size_directory, **parameters)
