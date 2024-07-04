import os
import sys
import json

import utils.utils

self = sys.modules[__name__]

import numpy as np
import pandas as pd
import tifffile
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("QtAgg")

import utils.utils as utils

import IO.IO as io


def read_data_group(filenames, combine=True, **args):
    """Turn a list of filenames for data into a numpy stack"""

    # check if stack already:
    if isinstance(filenames, np.ndarray):
        return filenames

    # read the individual files
    group = []
    for f in filenames:
        data = io.read(f, **args)
        data = np.reshape(data, (1,) + data.shape)
        group.append(data)

    if combine:
        return np.vstack(group)
    else:
        return group


def generate_average_maps(analysis_data_size_directory, **kwargs):
    stat_params = kwargs["statistics"]
    if stat_params["run_statistics"]:
        for channel in kwargs['study_params']['channels_to_segment']:
            for gn, gs in stat_params["groups"].items():
                group_sample_dirs = [os.path.join(analysis_data_size_directory, i)
                                     for i in os.listdir(analysis_data_size_directory) if i in gs]
                group_paths = [os.path.join(i, f'density_counts_{channel}.tif') for i in group_sample_dirs]
                group_data = read_data_group(group_paths)
                group_mean = np.mean(group_data, axis=0)
                io.write(os.path.join(analysis_data_size_directory, f'{gn}_average_{channel}.tif'),
                         group_mean)


def cutoff_p_values(pvals, p_cutoff=0.05):
    """cutt of p-values above a threshold.

  Arguments
  ---------
  p_valiues : array
    The p values to truncate.
  p_cutoff : float or None
    The p-value cutoff. If None, do not cut off.

  Returns
  -------
  p_values : array
    Cut off p-values.
  """
    pvals2 = pvals.copy()
    pvals2[pvals2 > p_cutoff] = p_cutoff
    return pvals2


def voxelize_stats(group1, group2, signed=False, remove_nan=True, p_cutoff=0.05, non_parametric=True):
    """Voxel-wise statistics between the individual voxels in group1 and group2

  Arguments
  ---------
  group1, group2 : array of arrays
    The group of voxelizations to compare.
  signed : bool
    If True, return also the direction of the changes as +1 or -1.
  remove_nan : bool
    Remove Nan values from the data.
  p_cutoff : None or float
    Optional cutoff for the p-values.

  Returns
  -------
  p_values : array
    The p values for the group wise comparison.
  """
    group1 = read_data_group(group1)
    group2 = read_data_group(group2)

    if non_parametric:  # Fixme: This is way too computationally heavy
        raise utils.CmliteError("Non-parametric voxel-wise statistics are not implemented yet!")
        # def mannwhitneyu_test_at_voxel(coords):
        #     x, y, z = coords
        #     print(x, y, z)
        #     sample1 = group1[:, x, y, z]
        #     sample2 = group2[:, x, y, z]
        #     stat, p = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
        #     return (x, y, z, stat, p)
        #
        # # Generate all voxel coordinates
        # all_voxel_coords = [(x, y, z) for x in range(268) for y in range(512) for z in range(369)]
        #
        # # Results list
        # results = []
        #
        # with ProcessPoolExecutor() as executor:
        #     # Map the mannwhitneyu_test_at_voxel function to all voxel coordinates
        #     futures = executor.map(mannwhitneyu_test_at_voxel, all_voxel_coords)
        #     for result in futures:
        #         results.append(result)
        # print(results)
        # # tvals, pvals = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    else:
        tvals, pvals = stats.ttest_ind(group1, group2, axis=0, equal_var=True)

    # remove nans
    if remove_nan:
        pi = np.isnan(pvals)
        pvals[pi] = 1.0
        tvals[pi] = 0

    pvals = cutoff_p_values(pvals, p_cutoff=p_cutoff)

    # return
    if signed:
        return pvals, np.sign(tvals)
    else:
        return pvals



def color_p_values(pvals, psign, positive=[1, 0], negative=[0, 1], p_cutoff=None,
                   positive_trend=[0, 0, 1, 0], negative_trend=[0, 0, 0, 1], pmax=None):
    pvalsinv = pvals.copy()
    if pmax is None:
        pmax = pvals.max()
    pvalsinv = pmax - pvalsinv

    if p_cutoff is None:  # color given p values

        d = len(positive)
        ds = pvals.shape + (d,)
        pvc = np.zeros(ds)

        # color
        ids = psign > 0
        pvalsi = pvalsinv[ids]
        for i in range(d):
            pvc[ids, i] = pvalsi * positive[i]

        ids = psign < 0
        pvalsi = pvalsinv[ids]
        for i in range(d):
            pvc[ids, i] = pvalsi * negative[i]

        return pvc

    else:  # split pvalues according to cutoff

        d = len(positive_trend)

        if d != len(positive) or d != len(negative) or d != len(negative_trend):
            raise RuntimeError(
                'colorPValues: postive, negative, postivetrend and negativetrend option must be equal length!')

        ds = pvals.shape + (d,)
        pvc = np.zeros(ds)

        pvals_cp = pvals.copy()
        ids = psign > 0
        non_zero = pvals > 0

        ##color
        # significant postive
        iii = np.logical_and(ids, non_zero)
        pvalsi = pvals_cp[iii]
        pvalsi[pvalsi < p_cutoff] = p_cutoff
        pvalsi = -np.log2(pvalsi)
        pvalsi = (pvalsi - (-np.log2(0.05))) / ((-np.log2(p_cutoff)) - (-np.log2(0.05)))
        w = positive
        for i in range(d):
            pvc[iii, i] = pvalsi * w[i]

        # significant negative
        iii = np.logical_and(~ids, non_zero)
        pvalsi = pvals_cp[iii]
        pvalsi[pvalsi < p_cutoff] = p_cutoff
        pvalsi = -np.log2(pvalsi)
        pvalsi = (pvalsi - (-np.log2(0.05))) / ((-np.log2(p_cutoff)) - (-np.log2(0.05)))
        w = negative
        for i in range(d):
            pvc[iii, i] = pvalsi * w[i]

        return pvc


def generate_pval_maps(analysis_data_size_directory, **kwargs):
    stat_params = kwargs["statistics"]
    for channel in kwargs['study_params']['channels_to_segment']:
        for comp in stat_params["group_comparisons"]:
            print(f"Running comparison for grp {comp[0]} vs grp {comp[1]}")

            print(f"Loading data for grp {comp[0]}")
            group_1 = stat_params["groups"][comp[0]]
            group_1_sample_dirs = [os.path.join(analysis_data_size_directory, i)
                                   for i in os.listdir(analysis_data_size_directory) if i in group_1]
            group_1_paths = [os.path.join(i, f'density_counts_{channel}.tif') for i in group_1_sample_dirs]
            group_1_data = np.array([tifffile.imread(i) for i in group_1_paths])
            group_1_data = [np.reshape(i, (1,) + i.shape) for i in group_1_data]
            group_1_data = np.vstack(group_1_data)

            print(f"Loading data for grp {comp[1]}")
            group_2 = stat_params["groups"][comp[1]]
            group_2_sample_dirs = [os.path.join(analysis_data_size_directory, i)
                                   for i in os.listdir(analysis_data_size_directory) if i in group_2]
            group_2_paths = [os.path.join(i, f'density_counts_{channel}.tif') for i in group_2_sample_dirs]
            group_2_data = np.array([tifffile.imread(i) for i in group_2_paths])
            group_2_data = [np.reshape(i, (1,) + i.shape) for i in group_2_data]
            group_2_data = np.vstack(group_2_data)

            # for pval in pval_cutoff:
            print(f"Running stats!")
            print(group_1_data.shape, group_2_data.shape)
            pvals, psign = voxelize_stats(group_1_data, group_2_data, signed=True,
                                          non_parametric=stat_params["voxel_wise"]["non_parametric"])
            print("Coloring p-values!")
            pvalscol = color_p_values(pvals, psign, positive=[0, 255, 0, 0], negative=[255, 0, 255, 0],
                                         p_cutoff=stat_params["voxel_wise"]["pval_cutoff"])
            tifffile.imwrite(os.path.join(analysis_data_size_directory,
                                          f'{comp[0]}_vs_{comp[1]}_pval_{channel}.tif'), pvalscol)

        if stat_params["inter_hemispherical_comparison"]:
            for gn, gs in stat_params["groups"].items():
                print(f"Running comparison for grp {gn} left vs right")

                print(f"Loading data for grp {gn}")
                group_sample_dirs = [os.path.join(analysis_data_size_directory, i)
                                     for i in os.listdir(analysis_data_size_directory) if i in gs]
                group_paths = [os.path.join(i, f'density_counts_{channel}.tif') for i in group_sample_dirs]
                group_data = np.array([tifffile.imread(i) for i in group_paths])
                group_data = [np.reshape(i, (1,) + i.shape) for i in group_data]
                group_data = np.vstack(group_data)

                left_right_axis = np.where(np.abs(np.array(kwargs["study_params"]["sample_permutation"])) == 1)[0][0]
                left_right_dim = group_data.shape[left_right_axis+1]
                left_right_mid = left_right_dim / 2
                if left_right_mid % 1 == 0:
                    group_data_right = group_data[:, :, :, :int(left_right_mid)]
                    group_data_left = group_data[:, :, :, int(left_right_mid):]
                else:
                    group_data_right = group_data[:, :, :, :int(np.ceil(left_right_mid))]
                    group_data_left = group_data[:, :, :, int(np.floor(left_right_mid)):]
                group_data_left = np.flip(group_data_left, left_right_axis+1)

                # for pval in pval_cutoff:
                pvals, psign = voxelize_stats(group_data_right, group_data_left, signed=True,
                                                      non_parametric=stat_params["voxel_wise"]["non_parametric"])
                pvalscol = color_p_values(pvals, psign, positive=[0, 255, 0, 0], negative=[255, 0, 255, 0],
                                             p_cutoff=stat_params["voxel_wise"]["pval_cutoff"])
                tifffile.imwrite(os.path.join(analysis_data_size_directory,
                                              f'{gn}_left_vs_right_pval_{channel}.tif'), pvalscol)


def extract_data(node, atlas_id, ids=None, names=None, acronyms=None, colors=None):
    if ids is None:
        ids = []
    if names is None:
        names = []
    if acronyms is None:
        acronyms = []
    if colors is None:
        colors = []

    ids.append(node['id'])
    names.append(node['name'])
    acronyms.append(node['acronym'])
    colors.append(node['color_hex_triplet'])

    for child in node.get('children', []):
        extract_data(child, atlas_id, ids, names, acronyms, colors)

    return ids, names, acronyms, colors


def run_region_wise_statistics(metadata_files, analysis_data_size_directory, **kwargs):
    stat_params = kwargs["statistics"]
    # Parse the JSON data
    with open(metadata, 'r') as file:
        data = json.load(file)
    # Process the data
    res_d = {}
    for msg in data['msg']:
        res_d["ids"], res_d["names"], res_d["acronyms"], res_d["colors"] = (extract_data(msg, msg['atlas_id']))

    if stat_params["region_wise"]["non_parametric"]:
        test_name = "mannwhitneyu"
    else:
        test_name = "ttest"

    for channel in kwargs['study_params']['channels_to_segment']:
        for comp in stat_params["group_comparisons"]:
            res_df = pd.DataFrame(data=res_d)
            res_df_shape = res_df.shape

            columns_to_create = [f"mean_grp_{comp[0]}", f"std_grp_{comp[0]}", f"sem_grp_{comp[0]}",
                                 f"mean_grp_{comp[1]}", f"std_grp_{comp[1]}", f"sem_grp_{comp[1]}",
                                 "pval"]
            for c in columns_to_create:
                res_df[c] = np.zeros(res_df_shape[0])

            all_group_occurences = []
            for group in comp:
                print(f"\nGetting region cell counts for group {group}!")
                group_occurences = []
                for animal in stat_params["groups"][group]:
                    print(f"Getting region cell counts for: {animal}!")
                    animal_directory = os.path.join(analysis_data_size_directory, animal)
                    cells_file_path = os.path.join(animal_directory, f"cells_transformed_{channel}.csv")
                    cells = pd.read_csv(cells_file_path, header=0, sep=';')
                    animal_occurences = []
                    for reg in res_df["names"]:
                        reg_oc = np.sum(cells[" name"] == reg)
                        animal_occurences.append(reg_oc)
                    res_df[animal] = animal_occurences
                    group_occurences.append(animal_occurences)
                group_occurences = np.array(group_occurences)
                all_group_occurences.append(group_occurences)
                res_df[f"mean_grp_{group}"] = np.mean(group_occurences, axis=0)
                res_df[f"std_grp_{group}"] = np.std(group_occurences, axis=0)
                res_df[f"sem_grp_{group}"] = stats.sem(group_occurences, axis=0)
            if stat_params["region_wise"]["non_parametric"]:
                res_df[f"pval"] = stats.mannwhitneyu(all_group_occurences[0],
                                                     all_group_occurences[1],
                                                     alternative='two-sided').pvalue
            else:
                res_df[f"pval"] = stats.ttest_ind(all_group_occurences[0],
                                                  all_group_occurences[1]).pvalue
            res_df = res_df.sort_values('pval', ascending=True)
            comparison_file_path = os.path.join(analysis_data_size_directory,
                                                f"{comp[0]}_vs_{comp[1]}_{test_name}_{channel}.csv")
            res_df.to_csv(comparison_file_path, index=False)

            comparison_data = pd.read_csv(comparison_file_path)
            acro = comparison_data["acronyms"]
            colors = comparison_data["colors"]

            log_2_fc = np.log2(comparison_data[f"mean_grp_{comp[0]}"] / comparison_data[f"mean_grp_{comp[1]}"])
            log_2_fc[log_2_fc == -np.inf] = 0
            log_2_fc[log_2_fc == np.inf] = 0
            log_pval = -np.log10(comparison_data["pval"])
            log_pval[log_2_fc == 0] = 0

            mask_up = np.logical_and(log_pval >= -np.log10(0.05), log_2_fc > 0)
            x_sign_up, y_sign_up, acro_up, color_up = (log_2_fc[mask_up], log_pval[mask_up], acro[mask_up],
                                                       colors[mask_up])
            color_up = np.array(color_up)
            color_up = np.array(['#' + color for color in color_up])
            mask_down = np.logical_and(log_pval >= -np.log10(0.05), log_2_fc < 0)
            x_sign_down, y_sign_down, acro_down, color_down = (log_2_fc[mask_down], log_pval[mask_down],
                                                               acro[mask_down], colors[mask_down])
            color_down = np.array(color_down)
            color_down = np.array(['#' + color for color in color_down])
            mask_ns = log_pval < -np.log10(0.05)
            x_sign_ns, y_sign_ns, acro_ns = log_2_fc[mask_ns], log_pval[mask_ns], acro[mask_ns]

            xlim = np.max(np.abs(log_2_fc))
            xlim = xlim + xlim * 0.1
            ylims = np.array([np.min(log_pval), np.max(log_pval)])
            ylim_range = np.abs(ylims[1] - ylims[0])

            # Volcano plots
            for i in range(2):
                fig = plt.figure()
                ax = plt.subplot(111)
                # ax.scatter(x_sign_up, y_sign_up, s=10, color="#FF5733") #one color
                ax.scatter(x_sign_up, y_sign_up, s=10, color=color_up)  # brain atlas color code
                # ax.scatter(x_sign_down, y_sign_down, s=10, color="#A7C7E7") #one color
                ax.scatter(x_sign_down, y_sign_down, s=10, color=color_down)  # brain atlas color code
                ax.scatter(x_sign_ns, y_sign_ns, s=10, color="gray")
                if i == 0:
                    for x, y, acro in zip(x_sign_up, y_sign_up, acro_up):
                        ax.text(x, y, acro, fontsize=7)
                    for x, y, acro in zip(x_sign_down, y_sign_down, acro_down):
                        ax.text(x, y, acro, fontsize=7)
                # ax.vlines(0, ylims[0] - ylim_range * 0.02, 3.5, color="black", linestyles="dashed")
                ax.vlines(0, ylims[0] - ylim_range * 0.02, ylims[1] + ylim_range * 0.1, color="black",
                          linestyles="dashed")
                # ax.set_xlim(-8, 8)
                ax.set_xlim(-xlim, xlim)
                # ax.set_ylim(ylims[0] - ylim_range * 0.02, 3.5)
                ax.set_ylim(ylims[0] - ylim_range * 0.02, ylims[1] + ylim_range * 0.1)
                ax.set_xlabel("log2(fold-change)", fontsize=12)
                ax.set_ylabel("-log10(pvalue)", fontsize=12)
                ax.set_title(f"{comp[0]} vs {comp[1]}", fontsize=12)
                plt.show()
                for ext in ["svg", "png"]:
                    if i == 0:
                        plt.savefig(os.path.join(analysis_data_size_directory,
                                                 f"{comp[0]}_vs_{comp[1]}_acro_{test_name}.{ext}"),
                                    dpi=300)
                    else:
                        plt.savefig(os.path.join(analysis_data_size_directory,
                                                 f"{comp[0]}_vs_{comp[1]}_{test_name}.{ext}"), dpi=300)