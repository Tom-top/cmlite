import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

# categories_n = categories[:-1]
# data_categories_n = merged_data[:-1]

categories_n = categories[:-1]
# data_categories_n = [merged_data[0][:2], merged_data[1][:11], merged_data[2][:34], merged_data[3][:77]]
data_categories_n = [merged_data[0][:1], merged_data[1][:6], merged_data[2][:12], merged_data[3][:28]]

# Initialize variables for the linkage matrix
Z = []
node_count = 0  # Start from 0 since we'll be assigning new indices to clusters
node_dict = {}

# Add initial clusters (first level) to the node_dict
for cls_idx, cls_d in data_categories_n[0].iterrows():
    cls_label = cls_d['Label']
    node_dict[cls_label] = node_count
    node_count += 1

# Process hierarchical clustering
for cls_idx, cls_d in data_categories_n[0].iterrows():
    cls_label = cls_d['Label']
    cls_count = cls_d["Count_df"]
    subcls_sum = 0
    subcls_idxs = []

    for subcls_idx, subcls_d in data_categories_n[1].iterrows():
        subcls_label = subcls_d['Label']
        subcls_count = subcls_d["Count_df"]

        if subcls_sum < cls_count:
            subcls_sum += subcls_count
            subcls_idxs.append(subcls_idx)
        else:
            break

        sptp_sum = 0
        sptp_idxs = []

        # Create a new node for the subclass and add to linkage
        if subcls_label not in node_dict:
            subcls_node = node_count
            Z.append([node_dict[cls_label], subcls_node, float(np.random.rand()), subcls_count])
            node_dict[subcls_label] = subcls_node
            node_count += 1
        else:
            subcls_node = node_dict[subcls_label]

        for sptp_idx, sptp_d in data_categories_n[2].iterrows():
            sptp_label = sptp_d['Label']
            sptp_count = sptp_d["Count_df"]

            if sptp_sum < subcls_count:
                sptp_sum += sptp_count
                sptp_idxs.append(sptp_idx)
            else:
                break

            cluster_sum = 0
            cluster_idxs = []

            # Create a new node for the supertype and add to linkage
            if sptp_label not in node_dict:
                sptp_node = node_count
                Z.append([node_dict[subcls_label], sptp_node, float(np.random.rand()), sptp_count])
                node_dict[sptp_label] = sptp_node
                node_count += 1
            else:
                sptp_node = node_dict[sptp_label]

            for cluster_idx, cluster_d in data_categories_n[3].iterrows():
                cluster_label = cluster_d['Label']
                cluster_count = cluster_d["Count_df"]

                if cluster_sum < sptp_count:
                    cluster_sum += cluster_count
                    cluster_idxs.append(cluster_idx)

                    # Create a new node for the cluster and add to linkage
                    if cluster_label not in node_dict:
                        cluster_node = node_count
                        Z.append([node_dict[sptp_label], cluster_node, float(np.random.rand()), cluster_count])
                        node_dict[cluster_label] = cluster_node
                        node_count += 1
                    else:
                        cluster_node = node_dict[cluster_label]
                else:
                    break

            # Drop the used cluster rows after the loop
            data_categories_n[3] = data_categories_n[3].drop(index=cluster_idxs)

        # Drop the used supertype rows after the loop
        data_categories_n[2] = data_categories_n[2].drop(index=sptp_idxs)

    # Drop the used subclass rows after the loop
    data_categories_n[1] = data_categories_n[1].drop(index=subcls_idxs)

# Convert the linkage list to a numpy array
Z = np.array(Z)

# Check for node usage issues
from collections import Counter
node_usage = Counter(Z[:, :2].flatten())
duplicates = [node for node, count in node_usage.items() if count > 1]
if duplicates:
    print(f"Warning: The following nodes are used more than once: {duplicates}")

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Cluster")
plt.ylabel("Distance")
plt.show()
