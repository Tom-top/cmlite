import os

from natsort import natsorted
import numpy as np
import pandas as pd

working_directory = "/mnt/data/Thomas/whole_brain/results/3d_views"

files_to_merge = [pd.read_csv(os.path.join(working_directory, i)) for i in natsorted(os.listdir(working_directory))
                  if i.startswith("cluster")]

data_merged = pd.concat(files_to_merge, axis=0)
df_concat_reset = data_merged.reset_index(drop=True)

df_concat_reset.to_feather(os.path.join(working_directory, "cluster_log2_mean_gene_expression_merge.feather"))

# df_concat_reset[df_concat_reset["cluster_name"] == "5006 RPA Pax6 Hoxb5 Gly-Gaba_1"]["ENSMUSG00000062995"]
