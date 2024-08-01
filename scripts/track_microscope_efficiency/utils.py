import os
import json
import logging

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib

matplotlib.use("Agg")

import utils.utils as ut


def extract_timestamps(working_directory, studies, scanning_system):
    """
    Extract timestamps, durations, valid scan percentages, and sample counts from the given directory and studies.

    :param working_directory: The base directory to search within.
    :param studies: The list of studies to analyze.
    :param scanning_system: The scanning system to match in the samples.
    :return: A list of timestamps, durations, a dictionary mapping dates to studies,
             a dictionary of valid scan percentages, and a dictionary of sample counts.
    """
    timestamps = []
    durations = []
    date_to_study = {}
    valid_scan_percentages = {}
    sample_counts = {}

    if not studies:
        studies = [i for i in os.listdir(working_directory) if i.startswith("24-") or i.startswith("23-")
                   or i.startswith("22-")]

    for folder in os.listdir(working_directory):
        if folder in studies:
            study_dir = os.path.join(working_directory, folder)
            keep_dir = os.path.join(study_dir, ".keep")
            if os.path.exists(keep_dir):
                ut.print_c(f"[INFO {folder}] Found .keep folder!")
                scan_summary_qc_file = os.path.join(keep_dir, "scan_summary_QC.csv")
                if os.path.exists(scan_summary_qc_file):
                    print("")
                    ut.print_c(f"[INFO {folder}] Found QC summary file: {scan_summary_qc_file}!")
                    scan_summary_qc = pd.read_csv(scan_summary_qc_file)

                    # ATTEMPT TO FIND QC SUMMARY IN U DRIVE
                    year_suffix = folder.split("-")[0]
                    year = "20" + str(year_suffix)
                    year_dir_u = os.path.join(r'U:\\', year)
                    if os.path.exists(year_dir_u):
                        if folder.endswith("bruker"):
                            study_dir_u = os.path.join(year_dir_u, "-".join(folder.split("-")[:-1]))
                        else:
                            study_dir_u = os.path.join(year_dir_u, folder)
                        if os.path.exists(study_dir_u):
                            scan_summary_qc_files = [os.path.join(study_dir_u, i) for i in os.listdir(study_dir_u)
                                                     if i.startswith("scan_summary") and i.endswith(".csv")]
                            # scan_summary_qc_file = os.path.join(study_dir_u, "scan_summary_QC_tto.csv")
                            if len(scan_summary_qc_files) >= 1:
                                ut.print_c(f"[INFO {folder}] Found {len(scan_summary_qc_files)} QC summary files!")
                                # scan_summary_qc_path = os.path.join(study_dir_u, scan_summary_qc_file)
                                # scan_summary_qc = pd.read_csv(scan_summary_qc_path)
                                scan_summary_qc = merge_csv_files(scan_summary_qc_files)

                                # Calculate the percentage of valid scans
                                for_analysis_col = scan_summary_qc["for analysis"]
                                all_scans = len(for_analysis_col)
                                valid_scans = np.sum(for_analysis_col == "x")
                                percent_valid_scans = (valid_scans / all_scans) * 100
                                valid_scan_percentages[folder] = percent_valid_scans

                                # Count the number of samples in the study
                                # sample_counts[folder] = all_scans  # That is not accurate: sum of scans and rescans
                                sample_counts[folder] = valid_scans
                            else:
                                valid_scan_percentages[folder] = None
                                sample_counts[folder] = None
                        else:
                            valid_scan_percentages[folder] = None
                            sample_counts[folder] = None

                    if os.path.isdir(study_dir):
                        raw_dir = os.path.join(study_dir, ".raw")
                        if os.path.exists(raw_dir):
                            for n, sample in enumerate(os.listdir(raw_dir)):
                                if not np.sum(scan_summary_qc["sample name"] == sample) == 0:
                                    scan_times = scan_summary_qc["scan time [secs]"][
                                        scan_summary_qc["sample name"] == sample]
                                    if len(scan_times) > 1:
                                        ut.print_c(
                                            f"[CRITICAL {folder}] Multiple scans detected under the same name: {len(scan_times)}!")
                                        scan_time = float(max(scan_times))
                                    else:
                                        scan_time = float(scan_times.iloc[0])
                                    durations.append(scan_time)
                                    sample_dir = os.path.join(raw_dir, sample)
                                    split_sample_name = sample.split("_")
                                    ss = [i for i in split_sample_name if i in ["M1", "M2", "M3", "M4"]]
                                    if len(ss) == 1:
                                        ss = ss[0]
                                    else:
                                        ut.print_c(f"[CRITICAL {folder}] Multiple scanning systems were detected!")
                                    if ss == scanning_system:
                                        if os.path.isdir(sample_dir):
                                            resolution_directory = [os.path.join(sample_dir, i) for i in
                                                                    os.listdir(sample_dir)
                                                                    if i.startswith("x")]
                                            if len(resolution_directory) == 1:
                                                resolution_directory = resolution_directory[0]
                                                config_task = [os.path.join(resolution_directory, i) for i in
                                                               os.listdir(resolution_directory)
                                                               if
                                                               i.startswith("config") and not i.endswith("merge.json")]
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
                                    ut.print_c(
                                        f"[WARNING {folder}] Sample: {sample} could not be located in the summary QC file!")

    return timestamps, durations, date_to_study, valid_scan_percentages, sample_counts


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
    # plt.show()
    plt.close()


def calculate_monthly_average_uptime(uptime_df, year):
    """
    Calculate the monthly average uptime percentage for a specific year, including months with no data.

    :param uptime_df: A DataFrame with dates, study names, and uptime percentages.
    :param year: The year for which to calculate the monthly averages.
    :return: A DataFrame with each month and its corresponding average uptime percentage.
    """
    # Ensure 'date' column is datetime
    uptime_df['date'] = pd.to_datetime(uptime_df['date'])

    # Filter the DataFrame for the selected year
    yearly_uptime_df = uptime_df[uptime_df['date'].dt.year == year]

    # If there is no data for the selected year, return an empty DataFrame with all months set to 0
    if yearly_uptime_df.empty:
        all_months = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='ME')
        return pd.DataFrame({'date': all_months, 'uptime_percentage': 0})

    # Group by year and month, then calculate the average uptime
    monthly_avg_uptime = yearly_uptime_df.groupby(yearly_uptime_df['date'].dt.to_period("M"))[
        'uptime_percentage'].mean().reset_index()

    # Convert period back to datetime for end of month
    monthly_avg_uptime['date'] = monthly_avg_uptime['date'].dt.to_timestamp('M')

    # Generate a DataFrame with all months of the selected year at the end of each month
    all_months = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='ME')
    all_months_df = pd.DataFrame({'date': all_months})

    # Merge with the actual data, filling missing months with zero uptime
    monthly_avg_uptime = pd.merge(all_months_df, monthly_avg_uptime, on='date', how='left').fillna(0)

    return monthly_avg_uptime


def plot_monthly_average_uptime(monthly_avg_uptime, scanning_system, year, saving_dir):
    """
    Plot the average monthly uptime percentage for a specific year and overlay a semi-transparent gray rectangle over future months.

    :param monthly_avg_uptime: A DataFrame with each month and its corresponding average uptime percentage.
    :param scanning_system: The scanning system used for the samples.
    :param year: The specific year for the plot.
    :param saving_dir: The directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))

    # Plot the uptime bars
    plt.bar(monthly_avg_uptime['date'].dt.strftime('%b'), monthly_avg_uptime['uptime_percentage'], color='skyblue',
            edgecolor='black')

    plt.ylim(0, 100)
    plt.title(f'Average Monthly Uptime for {scanning_system} in {year}')
    plt.xlabel('Month')
    plt.ylabel('Average Uptime Percentage (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Determine the current date and calculate the start and end of future months to gray out
    current_date = pd.Timestamp.now()
    future_months = monthly_avg_uptime[monthly_avg_uptime['date'] > current_date]

    if not future_months.empty:
        # Get the index of the first future month
        first_future_month_idx = future_months.index[0]

        # Get the x position of the first future month
        first_future_month_x = first_future_month_idx - 0.5

        # Get the width to cover the remaining months
        total_months = len(monthly_avg_uptime)
        width = total_months - first_future_month_idx

        # Add a semi-transparent rectangle over the future months area
        plt.gca().add_patch(Rectangle((first_future_month_x, 0), width, 100, color='gray', alpha=0.3))

    # Save the plot
    plt.savefig(os.path.join(saving_dir, f"{scanning_system}_monthly_average_uptime_{year}.png"), dpi=300)
    plt.show()


def plot_valid_scan_percentages(valid_scan_percentages, sample_counts, saving_dir):
    """
    Plot the percentage of valid scans for each study, with bars colored by the number of samples.
    Additionally, plot a horizontal line representing the average valid scan percentage.

    :param valid_scan_percentages: A dictionary with study names as keys and valid scan percentages as values.
    :param sample_counts: A dictionary with study names as keys and the number of samples as values.
    :param saving_dir: The directory where the plot will be saved.
    """
    # Convert the dictionaries to a DataFrame for easier plotting
    df = pd.DataFrame(list(valid_scan_percentages.items()), columns=['Study', 'Valid Scan Percentage'])

    # Add the sample counts to the DataFrame
    df['Sample Count'] = df['Study'].map(sample_counts)

    # Filter out studies where the percentage is NaN (i.e., where scan_summary_qc_file was not found)
    df = df.dropna()

    # Sort the DataFrame by valid scan percentage
    df = df.sort_values(by='Valid Scan Percentage', ascending=False)

    # Normalize the sample counts for coloring
    norm = mcolors.Normalize(df['Sample Count'].min(), df['Sample Count'].max())
    colors = cm.viridis(norm(df['Sample Count']))

    # Plot the data
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Study'], df['Valid Scan Percentage'], color=colors, edgecolor='black')
    plt.ylabel('Valid Scan Percentage (%)')
    plt.xlabel('Study')
    plt.title('Valid Scan Percentage by Study (Colored by Sample Count)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot the average line
    avg_value = df['Valid Scan Percentage'].mean()
    plt.axhline(y=avg_value, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_value:.2f}%')

    # Add a colorbar to show the sample count scale
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])  # For older versions of Matplotlib
    cbar = plt.colorbar(sm, ax=plt.gca())  # Associate the colorbar with the current axes
    cbar.set_label('Sample Count')

    # Add legend for the average line
    plt.legend()

    # Save the plot
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.savefig(os.path.join(saving_dir, "valid_scan_percentage_by_study.png"), dpi=300)
    # plt.show()
    plt.close()


def merge_csv_files(file_paths):
    """
    Merge multiple CSV files so that if a scan is labeled in the "for analysis" column with an "x",
    it keeps the "x" in the merged result, while ensuring only unique sample names are retained.

    :param file_paths: List of file paths to CSV files.
    :return: A merged DataFrame with unique sample names.
    """
    merged_df = pd.DataFrame()

    for file in file_paths:
        # print(f"Processing: {file}")
        # Read the CSV file
        df = pd.read_csv(file)

        # Standardize column names: remove leading/trailing spaces and lowercase all column names
        df.columns = df.columns.str.strip().str.lower()

        # Ensure consistent data types: convert 'for analysis' to string
        if 'for analysis' in df.columns:
            df['for analysis'] = df['for analysis'].fillna('').astype(str)

        # Check if 'sample name' column exists
        if 'sample name' not in df.columns:
            raise KeyError(f"'sample name' column not found in {file}")

        # Ensure all necessary columns are present (excluding 'comments' to avoid type issues)
        required_columns = ['sample name', 'for analysis', 'scan time [secs]']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''  # Add the missing column with empty values

        # Merge dataframes on relevant columns, ensuring that 'x' is retained in 'for analysis'
        merge_keys = [col for col in required_columns if col != 'comments']

        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=merge_keys, how='outer', suffixes=('', '_new'))

            # Safely combine the 'for analysis' columns
            if 'for analysis_new' in merged_df.columns:
                merged_df['for analysis'] = merged_df[['for analysis', 'for analysis_new']].apply(
                    lambda x: 'x' if 'x' in x.values else '', axis=1)
                merged_df = merged_df.drop(columns=['for analysis_new'])

    return merged_df
