import os

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

heatmap_directory = r"E:\tto\results\heatmaps\Zhuang-ABCA-1_old"
heatmaps = tifffile.imread(os.path.join(heatmap_directory, "expression_matrix_old.tif"))
gene_list = np.load(os.path.join(heatmap_directory, "gene_list_old.npy"))
n_genes = len(gene_list)
colormap = "hot"

mask_data = tifffile.imread(r"E:\tto\results\heatmaps\Zhuang-ABCA-1_old\Cartpt\analysis\Cartpt_mask.tif")

atlas_path = r"C:\Users\MANMONTALCINI\PycharmProjects\spatial_transcriptomics\gubra_and_ccfv3_alignment\gubra_template_coronal.tif"
atlas = tifffile.imread(atlas_path)
atlas_half = atlas[:, :, :184]
tissue_mask = atlas_half > 0

saving_folder = rf"E:\tto\results\heatmaps\Zhuang-ABCA-1_old\Cartpt\analysis"

mask_bin = mask_data == 255
#mask_bin = np.flip(mask_bin, 0)  # The mask is on the wrong side
mask_bin_half = mask_bin[:, :, :184]
mask_bin_half_sag = np.max(np.max(mask_bin_half, 1), 1)
s, e = np.where(np.diff(mask_bin_half_sag) == 1)[0]
mid = int((e + s)/2)

avg_gene_expression = []  # Average expression in the mask for each gene (uncorrected value)
hm_mid_planes = []
avg_norm_gene_expression = []  # Average normalized expression in the mask for each gene (normalized mean mask)

for n, (gene, hm) in enumerate(zip(os.listdir(heatmap_directory), heatmaps)):
    print(f"Computing average signal for: {gene} {n+1}/{n_genes}")
    avg_gene_exp_mask = np.mean(hm[mask_bin_half])
    avg_gene_expression.append(avg_gene_exp_mask)

    hm_mid_plane = hm[mid, :, :]
    hm_mid_planes.append(hm_mid_plane)

    tissue_not_mask = np.logical_and(tissue_mask, ~mask_bin_half)
    exp_brain = hm[tissue_not_mask]

    min_exp_brain, max_exp_brain = np.min(exp_brain), np.max(exp_brain)
    normalized_hm = (hm - min_exp_brain) / (max_exp_brain - min_exp_brain)

    avg_norm_gene_exp_mask = np.mean(normalized_hm[mask_bin_half])
    avg_norm_gene_expression.append(avg_norm_gene_exp_mask)

hm_mid_planes = np.array(hm_mid_planes)
n_hm_per_row = 10
n_top_hms = n_hm_per_row*5
n_rows = int(np.ceil(n_top_hms/n_hm_per_row))

####################################################################################################################
# Normalized expression
####################################################################################################################

avg_norm_gene_expression = np.array(avg_norm_gene_expression)

idx_sorted_gene_expression = np.argsort(avg_norm_gene_expression)
sorted_gene_expression = avg_norm_gene_expression[idx_sorted_gene_expression]
sorted_gene_list = gene_list[idx_sorted_gene_expression]

fig = plt.figure(figsize=(30, 10))
ax = plt.subplot(111)
ax.bar(np.arange(len(sorted_gene_expression[-50:])), sorted_gene_expression[-50:])
ax.set_xticks(np.arange(len(sorted_gene_expression[-50:])))
ax.set_xticklabels(sorted_gene_list[-50:], fontsize=8, rotation=45)
ax.set_ylabel("mean Log2(CPM+1) in region mask")
ax.set_title("Normalized expression levels")
plt.tight_layout()
plt.savefig(os.path.join(saving_folder, "normalized_mean_expression_50_top_genes.png"), dpi=300)
# plt.show()
