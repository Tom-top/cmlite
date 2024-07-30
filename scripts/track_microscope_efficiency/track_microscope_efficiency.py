import os
import json
import logging

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

matplotlib.use("Qt5Agg")

import utils.utils as ut


def extract_timestamps(working_directory, studies, scanning_system):
    """
    Extract timestamps and durations from the given directory and studies.

    :param working_directory: The base directory to search within.
    :param studies: The list of studies to analyze.
    :param scanning_system: The scanning system to match in the samples.
    :return: A list of timestamps, durations, and a dictionary mapping dates to studies.
    """
    timestamps = []
    durations = []
    date_to_study = {}

    if not studies:
        studies = [i for i in os.listdir(working_directory) if i.startswith("24-") or i.startswith("23-")]

    for folder in os.listdir(working_directory):
        if folder in studies:
            study_dir = os.path.join(working_directory, folder)
            keep_dir = os.path.join(study_dir, ".keep")
            if os.path.exists(keep_dir):
                print("")
                ut.print_c(f"[INFO {folder}] Found .keep folder!")
                scan_summary_qc_file = os.path.join(keep_dir, "scan_summary_QC.csv")
                if os.path.exists(scan_summary_qc_file):
                    ut.print_c(f"[INFO {folder}] Found QC summary file: {scan_summary_qc_file}!")
                    scan_summary_qc = pd.read_csv(scan_summary_qc_file)
                    # year_suffix = folder.split("-")[0]
                    # year = "20" + str(year_suffix)
                    # year_dir_u = os.path.join(r'U:\\', year)
                    # if os.path.exists(year_dir_u):
                    #     if folder.endswith("bruker"):
                    #         study_dir_u = os.path.join(year_dir_u, "-".join(folder.split("-")[:-1]))
                    #     else:
                    #         study_dir_u = os.path.join(year_dir_u, folder)
                    #     # scan_summary_qc_files = [i for i in os.listdir(study_dir_u) if i.startswith("scan_summary_QC")]
                    #     scan_summary_qc_file = os.path.join(study_dir_u, "scan_summary_QC_tto.csv")
                    #     print("")
                    #     if os.path.exists(scan_summary_qc_file):
                    #         ut.print_c(f"[INFO {folder}] Found QC summary file: {scan_summary_qc_file}")
                    #         scan_summary_qc_path = os.path.join(study_dir_u, scan_summary_qc_file)
                    #         scan_summary_qc = pd.read_csv(scan_summary_qc_path)
                    if os.path.isdir(study_dir):
                        raw_dir = os.path.join(study_dir, ".raw")
                        if os.path.exists(raw_dir):
                            for n, sample in enumerate(os.listdir(raw_dir)):
                                if not np.sum(scan_summary_qc["sample name"] == sample) == 0:
                                    scan_times = scan_summary_qc["scan time [secs]"][scan_summary_qc["sample name"] == sample]
                                    if len(scan_times) > 1:
                                        ut.print_c(f"[CRITICAL {folder}] Multiple scans detected under the same name: {len(scan_times)}!")
                                        scan_time = float(max(scan_times))
                                    else:
                                        scan_time = float(scan_times)
                                    durations.append(scan_time)
                                    sample_dir = os.path.join(raw_dir, sample)
                                    split_sample_name = sample.split("_")
                                    ss = [i for i in split_sample_name if i in ["M1", "M2", "M3", "M4"]]
                                    if len(ss) == 1:
                                        ss = ss[0]
                                    else:
                                        ut.print_c(f"[CRITICAL {folder}] Multiple scanning systems were detected!")
                                    # ss = sample.split("_")[-1]
                                    # if not ss.startswith("M"):
                                    #     ss = sample.split("_")[-2]
                                    if ss == scanning_system:
                                        if os.path.isdir(sample_dir):
                                            resolution_directory = [os.path.join(sample_dir, i) for i in os.listdir(sample_dir)
                                                                    if i.startswith("x")]
                                            if len(resolution_directory) == 1:
                                                resolution_directory = resolution_directory[0]
                                                config_task = [os.path.join(resolution_directory, i) for i in os.listdir(resolution_directory)
                                                               if i.startswith("config") and not i.endswith("merge.json")]
                                                if len(config_task) >= 1:
                                                    if n == 0:
                                                        ut.print_c(f"[INFO {folder}] Loading study data")
                                                    config_task = config_task[0]
                                                    config_task_data = ut.load_json_file(config_task)
                                                    task_tiles = config_task_data["input"]["image_file_paths"]["val"]
                                                    task_tile = task_tiles[0]
                                                    timestamp = task_tile.split("/")[7]
                                                    timestamps.append(timestamp)
                                                    date = datetime.strptime(timestamp, '%Y-%m-%d_%H%M%S').date()
                                                    if date not in date_to_study:
                                                        date_to_study[date] = {}
                                                    if folder not in date_to_study[date]:
                                                        date_to_study[date][folder] = 0
                                                    date_to_study[date][folder] += scan_time
                                    else:
                                        ut.print_c(f"[WARNING {folder}] Sample: {sample}. Scanning system is: {ss}!")
                                else:
                                    ut.print_c(f"[WARNING {folder}] Sample: {sample} could not be located in the summary QC file!")


    return timestamps, durations, date_to_study


def calculate_daily_uptime(date_to_study):
    """
    Calculate the daily uptime percentage for each study.

    :param date_to_study: A dictionary mapping dates to study names and their respective durations.
    :return: A DataFrame with dates, study names, and their uptime percentages.
    """
    daily_uptime = []
    for date, studies in date_to_study.items():
        total_uptime = sum(studies.values())
        for study, duration in studies.items():
            uptime_percentage = (duration / 86400) * 100
            if uptime_percentage > 100:
                uptime_percentage = 100
            daily_uptime.append({'date': date, 'study': study, 'uptime_percentage': uptime_percentage})

    # Convert to DataFrame
    uptime_df = pd.DataFrame(daily_uptime)
    return uptime_df


def generate_date_range(start_date, end_date):
    """
    Generate a range of dates from start_date to end_date.

    :param start_date: The start date.
    :param end_date: The end date.
    :return: A list of dates from start_date to end_date.
    """
    return pd.date_range(start=start_date, end=end_date).to_list()


def filter_uptime_df(uptime_df, time_frame, value):
    """
    Filter the uptime DataFrame based on the selected time frame.

    :param uptime_df: A DataFrame with dates, study names, and uptime percentages.
    :param time_frame: The time frame for filtering ('year', 'month', 'week').
    :param value: The specific year, month, or week to filter by.
    :return: A filtered DataFrame.
    """
    if time_frame == 'year':
        start_date = datetime(value, 1, 1)
        end_date = datetime(value, 12, 31)
    elif time_frame == 'month':
        start_date = datetime(value[0], value[1], 1)
        end_date = datetime(value[0], value[1], 1) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    elif time_frame == 'week':
        start_date = datetime.strptime(value + '-1', '%Y-%U-%w')
        end_date = start_date + pd.DateOffset(weeks=1) - pd.DateOffset(days=1)
    else:
        raise ValueError("Invalid time frame. Please choose from 'year', 'month', or 'week'.")

    # Create a date range for the specified time frame
    date_range = generate_date_range(start_date, end_date)

    # Ensure 'date' column is datetime
    uptime_df['date'] = pd.to_datetime(uptime_df['date'])

    # Filter the uptime data for the specified date range
    filtered_uptime_df = uptime_df[uptime_df['date'].isin(date_range)]

    # Ensure all dates in the range are present in the DataFrame
    for date in date_range:
        if date not in filtered_uptime_df['date'].values:
            filtered_uptime_df = pd.concat([filtered_uptime_df, pd.DataFrame([{'date': date}])])

    return filtered_uptime_df


def generate_random_color():
    """
    Generate a random color in hexadecimal format.

    :return: A random color string.
    """
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def plot_uptime(uptime_df, scanning_system, time_frame, value, saving_dir):
    """
    Plot the uptime percentage with bars colored based on the study.

    :param uptime_df: A DataFrame with dates, study names, and uptime percentages.
    :param scanning_system: The scanning system used for the samples.
    :param time_frame: The time frame for filtering ('year', 'month', 'week').
    :param value: The specific year, month, or week to filter by.
    """
    # Filter the DataFrame based on the selected time frame
    uptime_df['date'] = pd.to_datetime(uptime_df['date'])
    filtered_uptime_df = filter_uptime_df(uptime_df, time_frame, value)

    # Get unique studies
    studies = filtered_uptime_df['study'].dropna().unique()

    # Assign random colors to each study
    study_colors = {study: generate_random_color() for study in studies}

    # Pivot the DataFrame to get dates as index and studies as columns
    pivot_df = filtered_uptime_df.pivot(index='date', columns='study', values='uptime_percentage').fillna(0)

    # Plot the stacked bar chart
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(20, 6), color=[study_colors[study] for study in studies],
                  width=0.8, linewidth=0.5, edgecolor="black")

    # Highlight weekends
    for idx, date in enumerate(pivot_df.index):
        if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            ax.add_patch(Rectangle((idx - 0.5, 0), 1, 100, color='gray', alpha=0.2, edgecolor=None, linewidth=0))

    ax.set_ylim(0, 100)
    plt.title(f'Uptime {scanning_system}: {value}')
    plt.xlabel('Date')
    plt.ylabel('Uptime Percentage (%)')
    plt.xticks(rotation=45)
    if time_frame == 'year':
        month_positions = [(pivot_df.index.get_loc(
            pivot_df[pivot_df.index.month == month].index[len(pivot_df[pivot_df.index.month == month]) // 2])) for month
                           in range(1, 13) if len(pivot_df[pivot_df.index.month == month]) > 0]
        month_labels = [datetime(1900, month, 1).strftime('%b') for month in range(1, 13)]
        ax.set_xticks(month_positions)
        ax.set_xticklabels(month_labels)
    else:
        ax.set_xticklabels([date.strftime('%d-%b') for date in pivot_df.index], rotation=45)
    # Update legend to ignore NaN
    handles, labels = ax.get_legend_handles_labels()
    valid_handles_labels = [(handle, label) for handle, label in zip(handles, labels) if label != "nan"]
    if valid_handles_labels:
        handles, labels = zip(*valid_handles_labels)
        ax.legend(handles=handles, labels=labels, title='Studies', fontsize=8)
    plt.savefig(os.path.join(saving_dir, f"{scanning_system}_overview_{time_frame}_{value}.png"), dpi=300)
    plt.show()



# Example usage
working_directory = r'G:\\'  # Replace with the path to your directory
studies_to_analyze = []  # Replace with your actual study names
saving_dir = r"/default/path"  # PERSONAL
scanning_systems = ["M3", "M4"]  # Replace with your actual scanning system
#"M1", "M2",

for scanning_system in scanning_systems:
    timestamps, durations, date_to_study = extract_timestamps(working_directory, studies_to_analyze, scanning_system)
    # Ensure there are enough timestamps to calculate uptime
    if len(timestamps) > 1:
        uptime_df = calculate_daily_uptime(date_to_study)
        time_frame = 'year'  # Change to 'year', 'month', or 'week' as needed
        value = (2024)  # Change to the specific year, month, or week as needed
        plot_uptime(uptime_df, scanning_system, time_frame, value, saving_dir)
    else:
        print("Not enough timestamps to calculate uptime.")
