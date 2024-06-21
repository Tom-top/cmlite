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
import analysis.statistics as stats


# PATH PARAMETERS
user = "Thomas"
experiment = "test_exp_0"
run_statistics = False

# SAMPLE PARAMETERS
parameters = dict(samples_to_process=["sample_0"],
                  re_process=False,
                  scanning_system="zeiss",
                  scanning_pattern="z",
                  channels_to_stitch=[0, 1],
                  channels_to_segment=[1],
                  autofluorescence_channel=0,
                  animal_species="mouse",
                  atlas_to_use="gubra",
                  overwrite_results=False,
                  stitching=dict(search_params=np.array([20, 20, 5]),
                                 z_subreg_alignment=np.array([550, 650]),
                                 ),
                  cell_detection=dict(shape_detection=700,
                                      thresholds=dict(source=None,
                                                      size=(20, 500),
                                                      ),
                                      ),
                  voxelization=dict(radius=(7, 7, 7)),
                  statistics=dict(run_statistics=False,
                                  non_parametric=False,
                                  )
                  )

# CREATE ESSENTIAL DIRECTORIES
working_directory, raw_directory, analysis_directory = ut.create_ws(user, experiment)

# UNZIP AND GENERATE ATLAS/TEMPLATE FILES IN THE CORRECT ORIENTATION
annotation_file, reference_file = ano.prepare_annotation_files(
    annotation_file=os.path.join("resources/atlas",
                                 f"{parameters['atlas_to_use']}_annotation_{parameters['animal_species']}.tif"),
    reference_file=os.path.join("resources/atlas",
                                f"{parameters['atlas_to_use']}_reference_{parameters['animal_species']}.tif"),
    orientation=(3, -2, -1),
    )

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
#
# groups = dict(Grp1=["cfosbrain1", "cfosbrain2", "cfosbrain3"],
#               Grp2=["cfosbrain4", "cfosbrain5", "cfosbrain6", "cfosbrain7", "cfosbrain8", "cfosbrain9"],
#               Grp3=["cfosbrain10", "cfosbrain11", "cfosbrain12", "cfosbrain13", "cfosbrain14", "cfosbrain15"])
#
# if parameters["statistics"]["run_statistics"]:
#
#     for group_name, group in groups.items():
#         group_paths = [os.path.join(data_dir, f'{x}/density_counts.tif') for x in os.listdir(data_dir) if x in group and
#                        os.path.exists(os.path.join(data_dir, f'{x}/density_counts.tif'))]
#
#         group_data = stats.read_data_group(group_paths)
#         group_mean = np.mean(group_data, axis=0)
#         io.write(os.path.join(data_dir, f'{group_name}.tif'),
#                      horizontal_to_coronal(group_mean, bulbs_down=True, ventral_first=True))
