import os

import numpy as np
import pandas as pd

path_to_mean_gene_expression_file = r"cluster_log2_mean_gene_expression_merge_4.csv"  # Personal
path_to_mean_gene_expression_file_2 = r"cluster_log2_mean_gene_expression_final.csv"  # Personal

data = pd.read_feather(path_to_mean_gene_expression_file)
data_2 = pd.read_feather(path_to_mean_gene_expression_file_2)

data_f = data
data_f_2 = data_2

data_final = pd.concat([data_f, data_f_2], axis=0)
df_concat_reset = data_final.reset_index(drop=True)

df_concat_reset.to_csv(r"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results"
                       r"\3d_views\cluster_log2_mean_gene_expression_merge_5.csv", index=False)

df_concat_reset.to_feather(r"E:\tto\spatial_transcriptomics_results\whole_brain_gene_expression\results"
                           r"\3d_views\cluster_log2_mean_gene_expression_merge_5.feather")
