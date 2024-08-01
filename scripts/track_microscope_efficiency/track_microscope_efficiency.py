import scripts.track_microscope_efficiency.utils as ut

# Example usage
working_directory = r'G:\\'  # Replace with the path to your directory
studies_to_analyze = []  # Replace with your actual study names
saving_dir = r"/default/path"  # PERSONAL
scanning_systems = ["M3", "M4"]  # Replace with your actual scanning system
#"M1", "M2",

for scanning_system in scanning_systems:
    timestamps, durations, date_to_study, valid_scan_percentages, sample_counts = ut.extract_timestamps(
        working_directory,
        studies_to_analyze,
        scanning_system)
    ut.plot_valid_scan_percentages(valid_scan_percentages, sample_counts,
                                   saving_dir)  # Generate the plot for valid scan percentages
    # Ensure there are enough timestamps to calculate uptime
    if len(timestamps) > 1:
        uptime_df = ut.calculate_daily_uptime(date_to_study)
        time_frame = 'year'  # Change to 'year', 'month', or 'week' as needed
        value = (2024)  # Change to the specific year, month, or week as needed
        if isinstance(value, int):
            year = value
        else:
            year = value[0]
        monthly_avg_uptime = ut.calculate_monthly_average_uptime(uptime_df, year)  # Calculate monthly average uptime
        ut.plot_monthly_average_uptime(monthly_avg_uptime, scanning_system, year,
                                       saving_dir)  # Plot the monthly average uptime
        ut.plot_uptime(uptime_df, scanning_system, time_frame, value, saving_dir)
    else:
        print("Not enough timestamps to calculate uptime.")
