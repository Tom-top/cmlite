import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")

import utils.utils as ut

import spatial_transcriptomics.old.utils.plotting as st_plt

MAP_DIR = ut.create_dir(rf"/default/path")  # PERSONAL
MAIN_RESULTS_DIR = os.path.join(MAP_DIR, "results")
RESULTS_DIR = ut.create_dir(os.path.join(MAIN_RESULTS_DIR, "3d_views"))

ATLAS_USED = "gubra"
TRANSFORM_DIR = r"resources/abc_atlas"
REFERENCE_FILE = fr"resources/atlas/{ATLAS_USED}_reference_mouse.tif"
REFERENCE = tifffile.imread(REFERENCE_FILE)
REFERENCE_SHAPE = REFERENCE.shape
TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"hemisphere_mask.tif"))

# Specify the CSV file path
csv_file_path = os.path.join(RESULTS_DIR, "centers_of_mass_for_clusters.csv")
center_of_mass_data = pd.read_csv(csv_file_path)

# Selected cluster names (all will be plotted together)
defines_cluster_saving_name = "Drd1"
show_cluster_name = True
extension = "svg"
SHOW_REF = False

# defined_cluster_names = [i for i in center_of_mass_data["Cluster Name"] if "D1" in i]
defined_cluster_names = [
    '0943 STR D1 Gaba_1',
    '0944 STR D1 Gaba_1',
    '0945 STR D1 Gaba_1',
    '0946 STR D1 Gaba_1',
    '0947 STR D1 Gaba_1',
    '0948 STR D1 Gaba_2',
    '0949 STR D1 Gaba_2',
    '0950 STR D1 Gaba_3',
    '0951 STR D1 Gaba_3',
    '0952 STR D1 Gaba_3',
    '0953 STR D1 Gaba_4',
    '0954 STR D1 Gaba_4',
    '0955 STR D1 Gaba_5',
    '0956 STR D1 Gaba_5',
    '0957 STR D1 Gaba_5',
    '0958 STR D1 Gaba_6',
    '0959 STR D1 Gaba_6',
    '0960 STR D1 Gaba_7',
    '0961 STR D1 Gaba_8',
    '0962 STR D1 Gaba_8',
    '0963 STR D1 Gaba_8',
    '0964 STR D1 Gaba_9',
    '0990 STR D1 Sema5a Gaba_1',
    '0991 STR D1 Sema5a Gaba_1',
    '0992 STR D1 Sema5a Gaba_2',
    '0993 STR D1 Sema5a Gaba_2',
    '0994 STR D1 Sema5a Gaba_2',
    '0995 STR D1 Sema5a Gaba_2',
    '0996 STR D1 Sema5a Gaba_2',
    '0997 STR D1 Sema5a Gaba_3',
    '0998 STR D1 Sema5a Gaba_3',
    '0999 STR D1 Sema5a Gaba_4',
    '1000 STR D1 Sema5a Gaba_4'
]

if defined_cluster_names:
    # Filter the DataFrame to keep only rows with cluster names in the predefined list
    center_of_mass_data = center_of_mass_data[center_of_mass_data['Cluster Name'].isin(defined_cluster_names)]

########################################################################################################################
# PLOT ALL THE CLUSTERS
########################################################################################################################

cell_size = 20
cell_size_global = 20

SAVING_DIR = ut.create_dir(os.path.join(RESULTS_DIR, defines_cluster_saving_name))

########################################################################################################
# HORIZONTAL 3D VIEW
########################################################################################################
ori = "horizontal"
orix, oriy, mask_axis = 2, 0, 1
xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[1]

centers_of_mass = [center_of_mass_data['Center of Mass X'],
                   center_of_mass_data['Center of Mass Y'],
                   center_of_mass_data['Center of Mass Z']]
centers_of_mass = np.array(centers_of_mass).T

# Only neurons, class colors, all experiments
st_plt.plot_cells(centers_of_mass, REFERENCE, TISSUE_MASK, cell_colors=np.array(center_of_mass_data['Cluster Color']),
                  cell_categories=np.array(center_of_mass_data['Cluster Name']), non_neuronal_mask=None, xlim=xlim,
                  ylim=ylim,
                  orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                  sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"{defines_cluster_saving_name}_{ori}.{extension}"),
                  relevant_categories=[], show_outline=False, zoom=False,
                  show_ref=SHOW_REF, show_cluster_name=show_cluster_name)

########################################################################################################
# SAGITTAL 3D VIEW
########################################################################################################

ori = "sagittal"
orix, oriy, mask_axis = 0, 1, 2
xlim, ylim = REFERENCE_SHAPE[1], REFERENCE_SHAPE[2]

# Only neurons, class colors, all experiments
st_plt.plot_cells(centers_of_mass, REFERENCE, TISSUE_MASK, cell_colors=np.array(center_of_mass_data['Cluster Color']),
                  cell_categories=np.array(center_of_mass_data['Cluster Name']), non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                  orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                  sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"{defines_cluster_saving_name}_{ori}.{extension}"),
                  relevant_categories=[], show_outline=False, zoom=False,
                  show_ref=SHOW_REF, show_cluster_name=show_cluster_name)

########################################################################################################
# CORONAL 3D VIEW
########################################################################################################

ori = "coronal"
orix, oriy, mask_axis = 2, 1, 0  # Projection = 1
xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[2]

# Only neurons, class colors, all experiments
st_plt.plot_cells(centers_of_mass, REFERENCE, TISSUE_MASK, cell_colors=np.array(center_of_mass_data['Cluster Color']),
                  cell_categories=np.array(center_of_mass_data['Cluster Name']), non_neuronal_mask=None, xlim=xlim,
                  ylim=ylim,
                  orix=orix, oriy=oriy, orip=oriy, ori=ori, mask_axis=mask_axis, s=cell_size,
                  sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"{defines_cluster_saving_name}_{ori}.{extension}"),
                  relevant_categories=[], show_outline=False, zoom=False,
                  show_ref=SHOW_REF, show_cluster_name=show_cluster_name)
