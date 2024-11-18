"""
This script is designed to provide the gene expression of all cells in the ABC-atlas across specific clusters.
"""

import os

import numpy as np

ATLAS_USED = "gubra"
DATASETS = np.arange(1, 6, 1)
N_DATASETS = len(DATASETS)
CATEGORY_NAMES = ["neurotransmitter", "class", "subclass", "supertype", "cluster"]
NON_NEURONAL_CELL_TYPES = ["Astro", "Oligo", "Vascular", "Immune", "Epen", "OEC"]

gene = "Prkcd"  # "Dlk1"
