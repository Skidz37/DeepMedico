import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_signal(path, rate):
    """
    Loads signal data from a specified path, assuming a 'timestamp; value' format.
    It calculates the time index based on the sample rate.

    Args:
        path (str): The file path to the signal data.
        rate (int): The sample rate of the signal (samples per second).

    Returns:
        pd.DataFrame: A DataFrame with the signal values indexed by time (timedelta).
    """
    # Read the data, explicitly defining two columns ('timestamp_str' and 'value_str').
    # This prevents pandas from inferring a different number of columns
    # if the initial lines are malformed or different, which can cause ParserErrors.
    data = pd.read_csv(path, header=None, sep=';', names=['timestamp_str', 'value_str'], engine='python')

    # Cleanup: Convert the 'value_str' column to numeric, coercing errors to NaN, then drop NaNs.
    # Reset index to ensure a clean integer index for time calculation.
    numeric_data = pd.to_numeric(data['value_str'], errors='coerce').dropna().reset_index(drop=True)

    # Convert the Series back to a DataFrame and explicitly name the column '0'.
    # This ensures that subsequent access with `df[0]` works as expected.
    numeric_data = numeric_data.to_frame(name=0)

    # Calculate time based on the index and sample rate.
    # The index represents the sample number, so index / rate gives time in seconds.
    time = pd.to_timedelta(numeric_data.index / rate, unit='s')
    numeric_data['time'] = time
    numeric_data.set_index('time', inplace=True) # Set the calculated time as the DataFrame index
    return numeric_data

def plot(participant_path, name):
    """
    Loads and plots nasal airflow, thoracic movement, and SpO₂ signals,
    along with event markers if 'Flow Events.txt' is present.

    Args:
        participant_path (str): The directory path containing the participant's data files.
        name (str): The name of the participant, used for plot title and filename.
    """
    # Load the physiological signals using the corrected load_signal function
    nasal = load_signal(os.path.join(participant_path, 'Flow.txt'), 32)
    thoracic = load_signal(os.path.join(participant_path, 'Thorac.txt'), 32)
    spo2 = load_signal(os.path.join(participant_path, 'SPO2.txt'), 4)
    
    # Create a figure and a set of subplots (3 rows, 1 column)
    # sharex=True ensures that all subplots share the same x-axis for easy comparison.
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True) 
    
    # Plotting signals on their respective subplots
    # Convert time index to total seconds and then to minutes for the x-axis.
    axs[0].plot(nasal.index.total_seconds() / 60, nasal[0], label="Nasal Airflow")
    axs[0].set_title("Nasal Airflow Over Time")
    axs[0].set_ylabel("Airflow")
    axs[0].legend()

    axs[1].plot(thoracic.index.total_seconds() / 60, thoracic[0], label="Thoracic Movement", color='g')
    axs[1].set_title("Thoracic Movement Over Time")
    axs[1].set_ylabel("Movement")
    axs[1].legend()

    axs[2].plot(spo2.index.total_seconds() / 60, spo2[0], label="SpO₂", color='r')
    axs[2].set_title("SpO₂ Over Time")
    axs[2].set_xlabel("Time (minutes)")
    axs[2].set_ylabel("SpO₂ (%)")
    axs[2].legend()
    
    # --- Event Plotting Section ---
    events_file_path = os.path.join(participant_path, 'Flow Events.txt')
    if os.path.exists(events_file_path):
        print(f"Loading event data from {events_file_path}")
        # Read the event data, assuming it's a text file with flexible delimiters
        # and specific column names: 'event', 'start', 'end'.
        events = pd.read_csv(
            events_file_path, 
            header=None, 
            sep=r'\s*,\s*|\s+', # Flexible separator for spaces, tabs, or commas
            names=['event', 'start', 'end'], 
            engine='python' # Required for complex regex separators
        )

        # Cleanup: Ensure 'start' and 'end' columns are numeric and convert them to timedelta objects.
        events['start'] = pd.to_numeric(events['start'], errors='coerce')
        events['end'] = pd.to_numeric(events['end'], errors='coerce')
        events = events.dropna(subset=['start', 'end']) # Drop rows where start or end times are missing
        events['start'] = pd.to_timedelta(events['start'], unit='s')
        events['end'] = pd.to_timedelta(events['end'], unit='s')

        # Plotting events as vertical spans on all subplots
        apnea_label_added = False
        other_event_label_added = False

        for i, row in events.iterrows():
            event_type = str(row['event']).lower()
            event_color = 'orange' if event_type == 'apnea' else 'purple'
            event_label = 'Apnea Event' if event_type == 'apnea' else 'Other Event'
            
            for ax in axs:
                current_label = ""
                if event_type == 'apnea' and not apnea_label_added:
                    current_label = event_label
                    apnea_label_added = True
                elif event_type != 'apnea' and not other_event_label_added:
                    current_label = event_label
                    other_event_label_added = True
                
                ax.axvspan(
                    row['start'].total_seconds() / 60, # Convert start time to minutes
                    row['end'].total_seconds() / 60,   # Convert end time to minutes
                    color=event_color,
                    alpha=0.3, # Transparency of the shaded region
                    label=current_label # Add label only if it's the first instance of this type
                )
        
        # Consolidate and display legends for all subplots.
        for ax in axs:
            handles, labels = ax.get_legend_handles_labels()
            unique_labels_dict = dict(zip(labels, handles))
            unique_handles = list(unique_labels_dict.values())
            unique_labels = list(unique_labels_dict.keys())
            ax.legend(unique_handles, unique_labels, loc='best')
    else:
        print(f"Warning: 'Flow Events.txt' not found at {events_file_path}. Skipping event plotting.")


    # Add an overall title for the entire figure
    plt.suptitle(f"Physiological Signals for Participant: {name}", fontsize=16) 
    
    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
    
    # Create a directory for visualizations if it doesn't exist
    os.makedirs("Visualizations", exist_ok=True)
    
    # Save the plot as a PDF file
    plt.savefig(f"Visualizations/{name}.pdf")
    print(f"Saved Visualizations/{name}.pdf")
    # plt.show() # Uncomment if you want to display the plot interactively

def main():
    """
    Main function to parse command-line arguments and initiate plotting.
    """
    parser = argparse.ArgumentParser(description="Plot physiological signals and events.")
    parser.add_argument("-name", required=True, 
                        help="Path to the participant's data directory. "
                             "The plot will be saved with the base name of this path.")
    args = parser.parse_args()
    participant_name = os.path.basename(args.name) # Extract the base name for the plot title/filename
    plot(args.name, participant_name)

if __name__ == "__main__":
    main()
