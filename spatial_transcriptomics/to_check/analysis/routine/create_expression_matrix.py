import os

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")

heatmap_directory = r"/mnt/data/spatial_transcriptomics/expression_matrices"
n_genes = len(os.listdir(heatmap_directory))

atlas_path = r"/home/imaging/PycharmProjects/spatial_transcriptomics/gubra_and_ccfv3_alignment/gubra_template_coronal.tif"
atlas = tifffile.imread(atlas_path)
atlas_half = atlas[:, :, :184]
tissue_mask = atlas_half > 0

heatmaps = []
heatmaps_norm = []
gene_list = []

for n, gene in enumerate(os.listdir(heatmap_directory)):
    if os.path.isdir(os.path.join(heatmap_directory, gene)):
        print(f"Reading data for: {gene} {n+1}/{n_genes}")
        gene_list.append(gene)
        gene_directory = os.path.join(heatmap_directory, gene)
        gene_heatmap_path = os.path.join(gene_directory, f"{gene}_heatmap.tif")
        gene_heatmap = tifffile.imread(gene_heatmap_path)
        gene_heatmap_half = gene_heatmap[:, :, :184]
        heatmaps.append(gene_heatmap_half)

        exp_brain = gene_heatmap_half[tissue_mask]
        min_exp_brain, max_exp_brain = np.min(exp_brain), np.max(exp_brain)
        normalized_hm = (gene_heatmap - min_exp_brain) / (max_exp_brain - min_exp_brain)
        heatmaps_norm.append(normalized_hm)

        heatmaps.append(gene_heatmap_half)

heatmaps = np.array(heatmaps)
heatmaps_norm = np.array(heatmaps_norm)
gene_list = np.array(gene_list)
tifffile.imwrite(os.path.join(heatmap_directory, "expression_matrix.tif"), heatmaps)
tifffile.imwrite(os.path.join(heatmap_directory, "expression_matrix_norm.tif"), heatmaps_norm)
np.save(os.path.join(heatmap_directory, "gene_list.npy"), gene_list)