import os
from datetime import datetime

import utils.utils as ut
import scripts.track_microscope_efficiency.utils as mut

# Example usage
working_directory = r'G:\\'  # Replace with the path to your directory
studies_to_analyze = []  # Replace with your actual study names
current_date = datetime.now()
date_string = current_date.strftime('%y-%m-%d')
scanning_systems = ["M3", "M4"]  # Replace with your actual scanning system
time_frame = 'month'  # Change to 'year', 'month', or 'week' as needed
value = (2024, 9)  # Change to the specific year, month, or week as needed
saving_dir = ut.create_dir(fr"/default/path")  # PERSONAL
month_dir = ut.create_dir(os.path.join(saving_dir, "/default/path".join(value)))  # PERSONAL

for scanning_system in scanning_systems:
    saving_dir_scanning_system = ut.create_dir(os.path.join(saving_dir, scanning_system))
    timestamps, durations, date_to_study, valid_scan_percentages, sample_counts, performance_scores = \
        mut.extract_timestamps(
            working_directory,
            studies_to_analyze,
            scanning_system,
            saving_dir_scanning_system)
    performance_scores = {key: value for key, value in performance_scores.items() if value}
    mut.plot_performance_per_study(performance_scores, scanning_system, saving_dir_scanning_system)
    mut.plot_quality_per_study(valid_scan_percentages, sample_counts,
                               saving_dir_scanning_system)  # Generate the plot for valid scan percentages
    # Ensure there are enough timestamps to calculate uptime
    if len(timestamps) > 1:
        uptime_df = mut.calculate_daily_uptime(date_to_study)
        if isinstance(value, int):
            year = value
        else:
            year = value[0]
        monthly_avg_uptime = mut.calculate_monthly_average_uptime(uptime_df, year)  # Calculate monthly average uptime
        mut.plot_monthly_average_uptime(monthly_avg_uptime, scanning_system, year,
                                        saving_dir)  # Plot the monthly average uptime
        mut.plot_uptime(uptime_df, scanning_system, time_frame, value, saving_dir)
    else:
        print("Not enough timestamps to calculate uptime.")
