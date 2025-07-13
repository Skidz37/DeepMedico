import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def read_data(folder_path):
    nasal_airflow = pd.read_csv(os.path.join(folder_path, 'Flow.txt'), sep='\s+',skiprows=7,names=['timestamp','value'] ,parse_dates=['timestamp'],dayfirst=True)
    thoracic_movement = pd.read_csv(os.path.join(folder_path, 'Thorac.txt'), sep='\s+',skiprows=7,names=['timestamp', 'value'], parse_dates=['timestamp'], dayfirst=True)
    spo2 = pd.read_csv(os.path.join(folder_path, 'SPO2.txt'), sep=';', skiprows=6, names=['timestamp', 'value'], parse_dates=['timestamp'], dayfirst=True)
    events = pd.read_csv(os.path.join(folder_path, 'Flow Events.txt'), sep='\s+', parse_dates=['start_time', 'end_time'])
    return nasal_airflow, thoracic_movement, spo2, events

def resample_data(nasal_airflow, thoracic_movement, spo2):
    spo2_resampled = spo2.set_index('timestamp').resample('31.25ms').ffill().reset_index()
    return nasal_airflow, thoracic_movement, spo2_resampled

def plot_signals(nasal_airflow, thoracic_movement, spo2, events, output_path):
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axs[0].plot(nasal_airflow['timestamp'], nasal_airflow['value'], label='Nasal Airflow')
    axs[0].set_title('Nasal Airflow')
    axs[0].set_ylabel('Amplitude')
    axs[1].plot(thoracic_movement['timestamp'], thoracic_movement['value'], label='Thoracic Movement')
    axs[1].set_title('Thoracic Movement')
    axs[1].set_ylabel('Amplitude')
    axs[2].plot(spo2['timestamp'], spo2['value'], label='SpO2')
    axs[2].set_title('SpO2')
    axs[2].set_ylabel('Percentage')
    axs[2].set_xlabel('Time')
    for _, event in events.iterrows():
        for ax in axs:
            ax.axvspan(event['start_time'], event['end_time'], color='red', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def main(folder_path):
    nasal_airflow, thoracic_movement, spo2, events = read_data(folder_path)
    nasal_airflow, thoracic_movement, spo2 = resample_data(nasal_airflow, thoracic_movement, spo2)
    nasal_airflow_values = nasal_airflow['value'].to_numpy()
    thoracic_movement_values = thoracic_movement['value'].to_numpy()
    fs = 32  
    lowcut = 0.17
    highcut = 0.4
    nasal_airflow['value'] = apply_filter(nasal_airflow_values, lowcut, highcut, fs)
    thoracic_movement['value'] = apply_filter(thoracic_movement_values, lowcut, highcut, fs)
    output_path = os.path.join('Visualizations', f'{os.path.basename(folder_path)}_visualization.pdf')
    os.makedirs('Visualizations', exist_ok=True)
    plot_signals(nasal_airflow, thoracic_movement, spo2, events, output_path)
    print(f'Visualization saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate visualizations for sleep data.')
    parser.add_argument('-name', type=str, required=True, help='Folder path containing the signal files for one participant.')
    args = parser.parse_args()
    main(args.name)