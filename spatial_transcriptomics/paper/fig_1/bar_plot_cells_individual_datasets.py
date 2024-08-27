import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

working_dir = r"/default/path"  # PERSONAL
data_sheet_path = os.path.join(working_dir, "/default/path")  # PERSONAL

df = pd.read_excel(data_sheet_path)

# Bar plot for total number of cells in each dataset
# Define colors for individual datasets and for stacking the total
colors = ["#00FF2E", "#FF00D1", "#000FFF", "#FFF000", "black"]
stacked_colors = ["#00FF2E", "#FF00D1", "#000FFF", "#FFF000", "black"]

plt.figure(figsize=(8, 6))

# Plot individual dataset bars
for i in range(len(colors)):
    plt.bar(df['dataset'].astype(str)[i], df['all_cells_n'][i], color=colors[i],
            linewidth=0.5, edgecolor="black", width=0.8)

# Plot stacked bar for the total
total_height = 0
for i, color in enumerate(stacked_colors):
    plt.bar(df['dataset'].astype(str).iloc[-1], df.iloc[i, 1], bottom=total_height,
            color=color, linewidth=0.5, edgecolor="black", width=0.8)
    total_height += df.iloc[i, 1]

# Add title and labels
plt.title('Total Number of Cells in Each Dataset')
plt.xlabel('Dataset')
plt.ylabel('Number of Cells')
plt.savefig(os.path.join(working_dir, "cells_in_dataset.png"), dpi=300)
plt.savefig(os.path.join(working_dir, "cells_in_dataset.svg"), dpi=300)
plt.show()

# Bar plot for the proportion of neurons/non-neurons in each dataset
df['proportion_neurons'] = df['neurons_n'] / df['all_cells_n']
df['proportion_non_neurons'] = df['non_neurons_n'] / df['all_cells_n']

plt.figure(figsize=(8, 6))
index = df['dataset'].astype(str)

plt.bar(index, df['proportion_neurons'], width=0.8, linewidth=0.5,
        edgecolor="black", label='Neurons', color='black')
plt.bar(index, df['proportion_non_neurons'], width=0.8, linewidth=0.5,
        edgecolor="black", bottom=df['proportion_neurons'], label='Non-Neurons', color='white')

plt.title('Proportion of Neurons/Non-Neurons in Each Dataset')
plt.xlabel('Dataset')
plt.ylabel('Proportion')
plt.legend()
plt.savefig(os.path.join(working_dir, "proportion_neurons_in_dataset.png"), dpi=300)
plt.savefig(os.path.join(working_dir, "proportion_neurons_in_dataset.svg"), dpi=300)
plt.show()
