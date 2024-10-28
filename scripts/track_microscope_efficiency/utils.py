import os
import sys
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

# optimal_channel_times = {
#     "488": 100,  # MILLISECONDS PER PLANE (EXPOSURE)
#     "561": 100,  # MILLISECONDS PER PLANE (EXPOSURE)
#     "642": 100,  # MILLISECONDS PER PLANE (EXPOSURE)
#     "785": 1000,  # MILLISECONDS PER PLANE (EXPOSURE)
# }

# optimal_scan_times = {
#     "brain": {
#         "488": 0.5,  # HOURS
#         "561": 0.5,  # HOURS
#         "642": 1.5,  # HOURS
#         "785": 2,  # HOURS
#     },
#     "heart": {
#         "488": 0.5,  # HOURS
#         "561": 0.5,  # HOURS
#         "642": 1.5,  # HOURS
#         "785": 2,  # HOURS
#     },
#     "kidney": {
#         "488": 0.5,  # HOURS
#         "561": 0.5,  # HOURS
#         "642": 1.5,  # HOURS
#         "785": 2,  # HOURS
#     }
# }

optimal_scan_times = {
    "488": 0.5,  # HOURS
    "561": 0.5,  # HOURS
    "642": 1.5,  # HOURS
    "785": 2,  # HOURS
}

colors_per_channel = {
    "488": "#00f7ff",  # HOURS
    "561": "#c6ff00",  # HOURS
    "642": "#ff1600",  # HOURS
    "785": "#610000",  # HOURS
}

available_scanners = ["M1", "M2", "M3", "M4"]


def reset_logging(log_file_path):
    # Get the root logger
    logger = logging.getLogger()

    # Remove any existing handlers associated with the logger
    while logger.hasHandlers():
        logger.handlers[0].close()
        logger.removeHandler(logger.handlers[0])

    # Set up the new logging configuration (overwrite the old log file)
    logging.basicConfig(filename=log_file_path,
                        filemode='w',  # 'w' ensures the log file is overwritten each time
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def extract_timestamps(working_directory, studies, scanning_system, saving_dir):
    """
    Extract timestamps, durations, valid scan percentages, and sample counts from the given directory and studies.

    :param working_directory: The base directory to search within.
    :param studies: The list of studies to analyze.
    :param scanning_system: The scanning system to match in the samples.
    :return: A list of timestamps, durations, a dictionary mapping dates to studies,
             a dictionary of valid scan percentages, and a dictionary of sample counts.
    """

    # Reset logging for each function call
    log_file_path = os.path.join(saving_dir, 'output.log')
    reset_logging(log_file_path)

    # Example of how logging would now work after resetting the log file
    logging.info(f"Starting extraction for scanning system: {scanning_system}")

    timestamps = []
    durations = []
    date_to_study = {}
    valid_scan_percentages = {}
    performance_scores = {}
    sample_counts = {}

    if not studies:
        studies = [i for i in os.listdir(working_directory) if i.startswith("24-") or i.startswith("23-")
                   or i.startswith("22-")]

    for folder in os.listdir(working_directory):  # ITERATE OVER EVERY FOLDER IN NITROGEN
        print("")
        logging.info("")
        if folder in studies:  # IF THE FOLDER IS A STUDY
            study_dir = os.path.join(working_directory, folder)  # SET PATH TO THE STUDY DIRECTORY
            keep_dir = os.path.join(study_dir, ".keep")  # SET PATH TO THE KEEP DIRECTORY
            if os.path.exists(keep_dir):  # IF A KEEP DIRECTORY EXISTS
                ut.print_c(f"[INFO {folder}] Found .keep folder!")
                logging.info(f"[INFO {folder}] Found .keep folder!")

                ########################################################################################################
                # FIND THE SCAN SUMMARY QC FILE ON NITROGEN
                ########################################################################################################

                scan_summary_qc_file = os.path.join(keep_dir, "scan_summary_QC.csv")  # SET PATH TO SCAN SUMMARY QC FILE
                if os.path.exists(scan_summary_qc_file):  # IF THE SCAN SUMMARY FILE EXISTS
                    ut.print_c(f"[INFO {folder}] Found QC summary file: {scan_summary_qc_file}!")
                    logging.info(f"[INFO {folder}] Found QC summary file: {scan_summary_qc_file}!")
                    scan_summary_qc = pd.read_csv(scan_summary_qc_file)  # READ THE SCAN SUMMARY FILE

                    ####################################################################################################
                    # ATTEMPT TO FIND QC SUMMARY IN U DRIVE
                    ####################################################################################################

                    year_suffix = folder.split("-")[0]  # GET THE YEAR SUFFIX FROM STUDY FOLDER NAME
                    year = "20" + str(year_suffix)  # EXTRACT THE YEAR
                    year_dir_u = os.path.join(r'U:\\', year)  # SET PATH TO THE RELEVANT YEAR FOLDER ON U DRIVE
                    if os.path.exists(year_dir_u):  # IF THE YEAR FOLDER EXISTS
                        if folder.endswith("bruker"):  # IF THE STUDY WAS SCANNED ON BRUKER
                            # SET PATH TO THE STUDY FOLDER ON U DRIVE
                            study_dir_u = os.path.join(year_dir_u, "-".join(folder.split("-")[:-1]))
                            microscope = "bruker"
                        else:  # ELSE THE STUDY IS SCANNED ON LAVISION
                            study_dir_u = os.path.join(year_dir_u, folder)  # SET PATH TO THE STUDY FOLDER ON U DRIVE
                            microscope = "lavision"
                        if microscope == "bruker":
                            if os.path.exists(study_dir_u):  # IF THE STUDY FOLDER EXISTS ON U DRIVE
                                # FIXME: THE MERGING OF SCAN SUMMARY QC FILES SHOULD BE REMOVED
                                # FIND ALL SCAN SUMMARY QC FILES TO MERGE THEM
                                scan_summary_qc_files = [os.path.join(study_dir_u, i) for i in os.listdir(study_dir_u)
                                                         if i.startswith("scan_summary") and i.endswith(".csv")]
                                if len(scan_summary_qc_files) >= 1:  # IF AT LEAST 1 SCAN SUMMARY QC FILE WAS DETECTED
                                    ut.print_c(
                                        f"[INFO {folder}] Found {len(scan_summary_qc_files)} QC summary file(s)!")
                                    logging.info(
                                        f"[INFO {folder}] Found {len(scan_summary_qc_files)} QC summary file(s)!")
                                    scan_summary_qc = merge_csv_files(
                                        scan_summary_qc_files)  # MERGE THE SCAN SUMMARY QC FILES
                                    scan_summary_qc.to_csv(os.path.join(study_dir_u, "merged_scan_summary.csv"))

                                    # FIXME: THIS IS NOT ROBUST
                                    try:

                                        ########################################################################################
                                        #  GET THE SCANNING SYSTEM
                                        ########################################################################################

                                        sample_names = scan_summary_qc["sample name"]
                                        study_scanning_systems = [i.split("_")[-1] for i in sample_names]
                                        study_scanning_systems_mask = np.array([True if i == scanning_system else False
                                                                                for i in study_scanning_systems])

                                        ########################################################################################
                                        # CALCULATE THE NUMBER & PERCENTAGE OF VALID SCANS
                                        ########################################################################################

                                        # FETCH THE "FOR ANALYSIS" COLUMN INFO IN THE SCAN SUMMARY QC FILE
                                        for_analysis_col = scan_summary_qc["for analysis"][study_scanning_systems_mask]
                                        if len(for_analysis_col) > 0:
                                            all_scans = len(for_analysis_col)  # CALCULATE THE TOTAL NUMBER OF SCANS
                                            ut.print_c(f"[INFO {folder}] Found {all_scans} scans in total!")
                                            logging.info(f"[INFO {folder}] Found {all_scans} scans in total!")
                                            valid_scans = np.sum(
                                                for_analysis_col == "x")  # CALCULATE THE NUMBER OF SCANS SELECTED FOR ANALYSIS
                                            ut.print_c(f"[INFO {folder}] Found {valid_scans} valid scans!")
                                            logging.info(f"[INFO {folder}] Found {valid_scans} valid scans!")
                                            sample_counts[
                                                folder] = valid_scans  # SAVE THE NUMBER OF VALID SCANS IN DICT
                                            percent_valid_scans = (
                                                                              valid_scans / all_scans) * 100  # GET PERCENTAGE VALID SCANS
                                            valid_scan_percentages[
                                                folder] = percent_valid_scans  # SAVE THE PERCENTAGE OF VALID SCANS IN DICT
                                        else:
                                            valid_scan_percentages[
                                                folder] = None  # SAVE THE PERCENTAGE OF VALID SCANS IN DICT

                                        ########################################################################################
                                        # GET THE USED CHANNELS & METADATA
                                        ########################################################################################

                                        performance_scores[folder] = {}  # Initialize for each study (folder)

                                        for n, sample in scan_summary_qc.iterrows():  # ITERATE OVER EVERY SAMPLE
                                            sample_name = sample["sample name"]
                                            matches = [term for term in available_scanners if term in sample_name]
                                            if len(matches) == 1:
                                                sample_scanning_system = matches[0]
                                            else:
                                                ut.print_c(f"[WARNING {folder}] Sample: {sample}. "
                                                           f"No scanning system found: {matches}!")
                                                logging.info(
                                                    f"[WARNING {folder}] Sample: {sample}. "
                                                    f"No scanning system found: {matches}!")
                                                sample_scanning_system = ""
                                            if sample_scanning_system == scanning_system:
                                                performance_scores[folder][sample_name] = {}
                                                n_wavelengths = sample[
                                                    "n_wavelengths"]  # GET THE NUMBER OF WAVELENGTHS USED
                                                # n_channels = sample["n_channels"]
                                                excitation_wavelengths = [i for i in sample.keys() if i.startswith(
                                                    "excitation w")]  # FETCH ALL THE EXCITATION WAVELENGTH VALUES
                                                wavelengths_used = [sample[i] for i in excitation_wavelengths if
                                                                    not np.isnan(sample[i])]  # GET THE USED WAVELENGTHS
                                                wavelengths_used_mask = [True if not np.isnan(sample[i]) else False for
                                                                         i in
                                                                         excitation_wavelengths]  # GET THE USED WAVELENGTHS
                                                excitation_wavelengths_used = np.array(excitation_wavelengths)[
                                                    wavelengths_used_mask]
                                                n_wavelengths_used = len(
                                                    wavelengths_used)  # GET THE NUMBER OF USED WAVELENGTHS
                                                scanning_time = sample["scan time [secs]"]
                                                total_exposure = 0
                                                if n_wavelengths == n_wavelengths_used:  # IF THE NUMBER OF USED WAVELENGTHS MATCHED THE METADATA
                                                    for excitation_wavelength in excitation_wavelengths_used:
                                                        wavelength_value = excitation_wavelength.split(" ")[1]
                                                        try:
                                                            used_exposure = sample[f"exposure {wavelength_value} [ms]"]
                                                        except KeyError:
                                                            used_exposure = sample[f"exposure [ms]"]
                                                        total_exposure += used_exposure
                                                    for excitation_wavelength in excitation_wavelengths_used:
                                                        wavelength_value = excitation_wavelength.split(" ")[1]
                                                        try:
                                                            used_exposure = sample[f"exposure {wavelength_value} [ms]"]
                                                        except KeyError:
                                                            used_exposure = sample[f"exposure [ms]"]
                                                        relative_exposure = used_exposure / total_exposure
                                                        wavelength_used = sample[excitation_wavelength]
                                                        # target_exposure = optimal_channel_times[str(wavelength_used)]
                                                        # wavelength_performance_score = used_exposure/target_exposure
                                                        wavelength_scanning_time = relative_exposure * scanning_time
                                                        target_scanning_time = optimal_scan_times[
                                                                                   str(int(wavelength_used))] * 3600
                                                        wavelength_performance_score = target_scanning_time / wavelength_scanning_time
                                                        performance_scores[folder][sample_name][
                                                            str(int(wavelength_used))] = wavelength_performance_score
                                                else:  # IF THE NUMBER OF USED WAVELENGTHS DOES NOT MATCH THE METADATA
                                                    ut.print_c(
                                                        f"[WARNING {folder}] Number of wavelengths used does not match metadata!")
                                                    logging.info(
                                                        f"[WARNING {folder}] Number of wavelengths used does not match metadata!")
                                            else:
                                                ut.print_c(
                                                    f"[WARNING {folder}] {sample_name} was not scanned with"
                                                    f" {scanning_system} but {sample_scanning_system}!")
                                                logging.info(
                                                    f"[WARNING {folder}] {sample_name} was not scanned with"
                                                    f" {scanning_system} but {sample_scanning_system}!")
                                    except:
                                        ut.print_c(f"[WARNING {folder}] INVALID SCAN SUMMARY FILE")
                                        logging.info(
                                            f"[WARNING {folder}] INVALID SCAN SUMMARY FILE")
                                else:  # IF NO SCAN SUMMARY QC FILE WERE DETECTED
                                    valid_scan_percentages[folder] = None  # SET VALID SCAN PERCENTAGE TO NONE
                                    sample_counts[folder] = None  # SET SAMPLE COUNTS TO NONE
                            else:  # IF THE STUDY FOLDER DOES NOT EXIST ON U DRIVE
                                valid_scan_percentages[folder] = None  # SET VALID SCAN PERCENTAGE TO NONE
                                sample_counts[folder] = None  # SET SAMPLE COUNTS TO NONE
                    else:  # IF THE YEAR FOLDER DOES NOT EXIST
                        ut.CmliteError(f"[ERROR {folder}] Year folder {year} does not exist on U drive!")
                        logging.info(
                            f"[ERROR {folder}] Year folder {year} does not exist on U drive!")

                    if os.path.isdir(study_dir):
                        raw_dir = os.path.join(study_dir, ".raw")
                        if os.path.exists(raw_dir):
                            for n, sample in enumerate(os.listdir(raw_dir)):
                                if sample not in [".Data-sync-controlled.txt"]:
                                    sample_mask = np.array(scan_summary_qc["sample name"] == sample)
                                    if not np.sum(sample_mask) == 0:
                                        try:  # Fixme: CONSISTENCY!!
                                            scan_times = scan_summary_qc["scan time [secs]"][sample_mask]
                                        except KeyError:
                                            scan_times = scan_summary_qc["scan time [s]"][sample_mask]
                                        if len(scan_times) > 1:
                                            ut.print_c(
                                                f"[CRITICAL {folder}] Multiple scans detected under the same name: {len(scan_times)}!")
                                            logging.info(
                                                f"[CRITICAL {folder}] Multiple scans detected under the same name: {len(scan_times)}!")
                                            scan_time = float(max(scan_times))
                                        else:
                                            scan_time = float(scan_times.iloc[0])
                                        durations.append(scan_time)
                                        sample_dir = os.path.join(raw_dir, sample)
                                        ss = [term for term in available_scanners if term in sample]
                                        if len(ss) == 1:
                                            ss = ss[0]
                                        else:
                                            ut.print_c(f"[CRITICAL {folder}] Multiple scanning systems were detected!")
                                            logging.info(
                                                f"[CRITICAL {folder}] Multiple scanning systems were detected!")
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
                                                                   i.startswith("config") and not i.endswith(
                                                                       "merge.json")]
                                                    if len(config_task) >= 1:
                                                        if n == 0:
                                                            ut.print_c(f"[INFO {folder}] Loading study data")
                                                            logging.info(
                                                                f"[INFO {folder}] Loading study data")
                                                        config_task = config_task[0]
                                                        config_task_data = ut.load_json_file(config_task)
                                                        task_tiles = config_task_data["input"]["image_file_paths"][
                                                            "val"]
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
                                            ut.print_c(
                                                f"[WARNING {folder}] Sample: {sample}. Scanning system is: {ss}!")
                                            logging.info(
                                                f"[WARNING {folder}] Sample: {sample}. Scanning system is: {ss}!")
                                    else:
                                        ut.print_c(
                                            f"[WARNING {folder}] Sample: {sample} could not be located in the summary QC file!")
                                        logging.info(
                                            f"[WARNING {folder}] Sample: {sample} could not be located in the summary QC file!")
                                else:
                                    ut.print_c(f"[WARNING {folder}] Skipping file: {sample}!")
                                    logging.info(
                                        f"[WARNING {folder}] Skipping file: {sample}!")

    return timestamps, durations, date_to_study, valid_scan_percentages, sample_counts, performance_scores


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
        all_months = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='M')
        return pd.DataFrame({'date': all_months, 'uptime_percentage': 0})

    # Group by year and month, then calculate the average uptime
    monthly_avg_uptime = yearly_uptime_df.groupby(yearly_uptime_df['date'].dt.to_period("M"))[
        'uptime_percentage'].mean().reset_index()

    # Convert period back to datetime for end of month
    monthly_avg_uptime['date'] = monthly_avg_uptime['date'].dt.to_timestamp('M')

    # Generate a DataFrame with all months of the selected year at the end of each month
    all_months = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='M')
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


def plot_quality_per_study(valid_scan_percentages, sample_counts, saving_dir):
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


def plot_performance_per_study(performance_scores, scanning_system, saving_dir):
    """
    Plot the efficiency for each study. Each group of bars represents a study,
    and each bar within the group represents the average performance across samples for each channel.

    :param performance_scores: A dictionary with study names as keys and dictionaries of performance scores as values.
    :param saving_dir: The directory where the plot will be saved.
    """

    # Dictionary to store average performances for each study and each channel
    study_channel_performance = {}

    # Calculate average performance for each channel in each study
    for study, samples in performance_scores.items():
        channel_performance = {}

        # Aggregate performance scores by channel
        for sample, channels in samples.items():
            for channel, performance in channels.items():
                if channel not in channel_performance:
                    channel_performance[channel] = []
                channel_performance[channel].append(performance)

        # Calculate average performance for each channel
        avg_performance = {channel: np.mean(scores) for channel, scores in channel_performance.items()}
        study_channel_performance[study] = avg_performance

    # Get the list of all unique channels across all studies
    all_channels = sorted(set(channel for study in study_channel_performance.values() for channel in study.keys()))

    # Set the positions and width for the bars
    num_studies = len(study_channel_performance)
    num_channels = len(all_channels)
    bar_width = 1
    group_width = num_channels * bar_width + 0.5  # Increase space between groups of studies
    index = np.arange(num_studies) * group_width  # Adjust index for separation between study groups

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Plot bars for each channel within each study group
    for i, (study, avg_performance) in enumerate(study_channel_performance.items()):
        for j, channel in enumerate(all_channels):
            # Get the average performance for the current channel in the current study
            performance = avg_performance.get(channel, 0)
            # Set the position for the current bar
            position = index[i] + j * bar_width
            # Plot the bar with the specific color for each channel
            plt.bar(position, performance, bar_width, color=colors_per_channel[channel], edgecolor="black",
                    linewidth=0.5, alpha=0.5, label=f'{study} - {channel}' if i == 0 else "")

    # Set the labels and title
    plt.xlabel('Studies')
    plt.ylabel('Average Performance')
    plt.title('Average Performance per Channel for Each Study')

    # Set x-ticks to be centered on each study group
    plt.xticks(index + (num_channels - 1) * bar_width / 2, study_channel_performance.keys(), rotation=90)
    # plt.legend(title='Channels')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1)

    # Save the plot
    plt.tight_layout()
    # plt.savefig(f"{saving_dir}/efficiency_all_studies_plot_{scanning_system}.png", dpi=300)
    plt.savefig(f"{saving_dir}/efficiency_all_studies_plot.png", dpi=300)
    # plt.show()  # Uncomment to display the plot
    plt.close()


def load_qc_files(file):
    try:
        # Attempt to read with default delimiter (comma)
        df = pd.read_csv(file)

        # If the column count is still 1, try using semicolon as delimiter
        if len(df.columns) == 1:
            df = pd.read_csv(file, delimiter=';')

        # Standardize column names: remove leading/trailing spaces and lowercase all column names
        df.columns = df.columns.str.strip().str.lower()

        # Ensure consistent data types: convert 'for analysis' to string
        if 'for analysis' in df.columns:
            df['for analysis'] = df['for analysis'].fillna('').astype(str)

        # Check if 'sample name' column exists
        if 'sample name' not in df.columns:
            raise KeyError(f"'sample name' column not found in {file}")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load {file}: {e}")


def merge_csv_files(file_paths):
    """
    Merge multiple CSV files so that if a scan is labeled in the "for analysis" column with an "x",
    it keeps the "x" in the merged result, while ensuring only unique sample names are retained.

    :param file_paths: List of file paths to CSV files.
    :return: A merged DataFrame with unique sample names.
    """
    merged_df = pd.DataFrame()

    for file in file_paths:
        # Read the CSV file
        df = load_qc_files(file)

        # Ensure all necessary columns are present (excluding 'comments' to avoid type issues)
        required_columns = ['sample name', 'for analysis', 'scan time [secs]']
        for col in required_columns:
            if col not in df.columns:
                if col == 'scan time [secs]':
                    if 'scan time [s]' not in df.columns:
                        df[col] = ''
                else:
                    df[col] = ''  # Add the missing column with empty values

        # Merge dataframes on 'sample name' and retain all columns
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='sample name', how='outer', suffixes=('', '_new'))

            # Combine columns with the same name
            for col in df.columns:
                if col != 'sample name' and col + '_new' in merged_df.columns:
                    # If the column is 'for analysis', ensure 'x' is retained
                    if col == 'for analysis':
                        merged_df[col] = merged_df[[col, col + '_new']].apply(
                            lambda x: 'x' if 'x' in x.values else '', axis=1)
                    else:
                        # Merge other columns without data loss
                        merged_df[col] = merged_df[[col, col + '_new']].bfill(axis=1).iloc[:, 0]
                    # Drop the new column suffix
                    merged_df = merged_df.drop(columns=[col + '_new'])

    return merged_df
