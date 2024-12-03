import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
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
# TISSUE_MASK = tifffile.imread(os.path.join(MAP_DIR, r"hemisphere_mask.tif"))

# Specify the CSV file path
csv_file_path = os.path.join(TRANSFORM_DIR, "centers_of_mass_for_clusters.csv")
center_of_mass_data = pd.read_csv(csv_file_path)

# Selected cluster names (all will be plotted together)
defines_cluster_saving_name = "top_clusters"
show_cluster_name = False
color_by_enrichment = True
extension = "png"
SHOW_REF = True

sorted_and_enriched_clusters = pd.read_csv(os.path.join(MAP_DIR, "results/3d_views/cluster_labels_and_percentages.csv"))

# defined_cluster_names = [i for i in center_of_mass_data["Cluster Name"] if "D1" in i]
defined_cluster_names = list(center_of_mass_data["Cluster Name"])
cluster_names = list(sorted_and_enriched_clusters["Label"])
enrichment_coeff = np.array(sorted_and_enriched_clusters["Percentage"])

# Map Labels to Percentages
label_to_percentage = dict(zip(cluster_names, enrichment_coeff))
# Add a new Percentages column to the DataFrame
center_of_mass_data["Percentage"] = center_of_mass_data["Cluster Name"].map(label_to_percentage).fillna(0)

enrichment_coeff_all_clusters = np.array(center_of_mass_data["Percentage"])
# threshold_mask = enrichment_coeff_all_clusters > 40

clip_up = 100
enrichment_coeff_all_clusters[enrichment_coeff_all_clusters > clip_up] = clip_up
enrichment_coeff_all_clusters = ((enrichment_coeff_all_clusters - np.min(enrichment_coeff_all_clusters)) /
                                 (np.max(enrichment_coeff_all_clusters) - np.min(enrichment_coeff_all_clusters)))
# enrichment_coeff_all_clusters = [float(x*2.55) for x in enrichment_coeff_all_clusters]

if defined_cluster_names:
    # Filter the DataFrame to keep only rows with cluster names in the predefined list
    center_of_mass_data = center_of_mass_data[center_of_mass_data['Cluster Name'].isin(defined_cluster_names)]

if color_by_enrichment:
    # cmap = plt.cm.viridis_r
    cmap = plt.cm.gist_heat_r
    colors = cmap(enrichment_coeff_all_clusters)
else:
    colors = np.array(center_of_mass_data['Cluster Color'])
    # Remove the '#' if present
    hex_colors = [i.lstrip('#') for i in colors]
    # Extract the RGB values
    colors = np.array([(int(i[:2], 16)/255, int(i[2:4], 16)/255, int(i[4:], 16)/255, 1) for i in hex_colors])

########################################################################################################################
# PLOT ALL THE CLUSTERS
########################################################################################################################

cell_size = 2
cell_size_global = 2

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
st_plt.plot_cells(centers_of_mass, REFERENCE, TISSUE_MASK, cell_colors=colors,
                  cell_categories=np.array(center_of_mass_data['Cluster Name']), non_neuronal_mask=None, xlim=xlim,
                  ylim=ylim,
                  orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                  sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"{defines_cluster_saving_name}_{ori}.{extension}"),
                  relevant_categories=[], show_outline=False, zoom=False,
                  show_ref=SHOW_REF, show_cluster_name=show_cluster_name, plot_individual_categories=False,
                  surface_projection=False, max_projection=True)

########################################################################################################
# SAGITTAL 3D VIEW
########################################################################################################

ori = "sagittal"
orix, oriy, mask_axis = 0, 1, 2
xlim, ylim = REFERENCE_SHAPE[1], REFERENCE_SHAPE[2]

# Only neurons, class colors, all experiments
st_plt.plot_cells(centers_of_mass, REFERENCE, TISSUE_MASK, cell_colors=colors,
                  cell_categories=np.array(center_of_mass_data['Cluster Name']), non_neuronal_mask=None, xlim=xlim, ylim=ylim,
                  orix=orix, oriy=oriy, orip=orix, ori=ori, mask_axis=mask_axis, s=cell_size,
                  sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"{defines_cluster_saving_name}_{ori}.{extension}"),
                  relevant_categories=[], show_outline=False, zoom=False,
                  show_ref=SHOW_REF, show_cluster_name=show_cluster_name, plot_individual_categories=False,
                  surface_projection=False, max_projection=True)

########################################################################################################
# CORONAL 3D VIEW
########################################################################################################

ori = "coronal"
orix, oriy, mask_axis = 2, 1, 0  # Projection = 1
xlim, ylim = REFERENCE_SHAPE[0], REFERENCE_SHAPE[2]

# Only neurons, class colors, all experiments
st_plt.plot_cells(centers_of_mass, REFERENCE, TISSUE_MASK, cell_colors=colors,
                  cell_categories=np.array(center_of_mass_data['Cluster Name']), non_neuronal_mask=None, xlim=xlim,
                  ylim=ylim,
                  orix=orix, oriy=oriy, orip=oriy, ori=ori, mask_axis=mask_axis, s=cell_size,
                  sg=cell_size_global, saving_path=os.path.join(SAVING_DIR, f"{defines_cluster_saving_name}_{ori}.{extension}"),
                  relevant_categories=[], show_outline=False, zoom=False,
                  show_ref=SHOW_REF, show_cluster_name=show_cluster_name, plot_individual_categories=False,
                  surface_projection=False, max_projection=True)
