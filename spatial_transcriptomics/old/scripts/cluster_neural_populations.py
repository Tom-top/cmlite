import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Paths and Constants
TRANSFORM_DIR = "resources/abc_atlas"
CLUSTER_CSV_PATH = os.path.join(TRANSFORM_DIR, "centers_of_mass_for_clusters.csv")
MEAN_EXPR_PATH = os.path.join(TRANSFORM_DIR, "cluster_log2_mean_gene_expression_merge.feather")
CLUSTER_LABELS_PATH = "/mnt/data/Thomas/mPD5/results/3d_views/cluster_labels_and_percentages.csv"
CLUSTER_OVERLAP_THRESHOLD = 40

# Load cluster overlap data
sorted_and_enriched_clusters = pd.read_csv(CLUSTER_LABELS_PATH)
threshold_mask = sorted_and_enriched_clusters["Percentage"] > CLUSTER_OVERLAP_THRESHOLD
cluster_names = sorted_and_enriched_clusters["Label"][threshold_mask]

# Load gene expression data
mean_expression_matrix = pd.read_feather(MEAN_EXPR_PATH)

# Extract gene expression data for selected clusters
gene_expression_data = pd.concat(
    [
        mean_expression_matrix[mean_expression_matrix["cluster_name"] == cluster_name]
        for cluster_name in cluster_names
    ]
)

# Drop non-numeric columns (e.g., "cluster_name")
gene_expression_values = gene_expression_data.drop(columns=["cluster_name"]).select_dtypes(include=[np.number])

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(gene_expression_values)

# Perform hierarchical clustering
linkage_matrix = linkage(scaled_data, method="ward")

# Plot the dendrogram
plt.figure(figsize=(10, 8))
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()
