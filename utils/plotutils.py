import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
# from helper import *
from utils.helper import *


def plot_target_with_panel_changes(df):
    # Convert the 'time' column to pandas datetime format
    df['time'] = pd.to_datetime(df['time'])

    # Discretize the 'panel_number' column and convert it to a categorical variable
    # df['panel_number'] = pd.Categorical(df['panel_number'])

    # Get unique panel numbers
    unique_panels = df['panel_number'].unique()

    # Create a time series plot for each panel number
    for panel_code in unique_panels:
        panel_df = df[df['panel_number'] == panel_code]
        plt.plot(panel_df['time'], panel_df['target'], label=f'Panel {panel_code}', marker='o', alpha=0.3)

    # Add vertical lines at the 'panel_number' change points
    # panel_change_indices = df.index[df['panel_number'].diff() != 0]
    # for idx in panel_change_indices:
    #     plt.axvline(x=df['time'].iloc[idx], color='gray', linestyle='--', alpha=0.5)

    # Add labels and legend
    plt.xlabel('Time')
    plt.ylabel('Target Value')
    plt.title('Target Time Series with Panel Number Changes')
    plt.legend()

    # Show the plot
    plt.show()


def plot_time_series_for_one_panel(df, sample_rate=1000, panel=0, filter=False):
    """
    Plot time series data in separate panels.

    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data.
            The DataFrame should have a column containing timestamps with a name containing 'time',
            and one or more columns containing the time series data.

    Returns:
        None (Plots the time series panels using matplotlib).
        :param panel:
        :param sample_rate:
        :param filter:
    """
    p_no = df['panel_number'].unique()[panel]
    df = df[df['panel_number'] == p_no]
    df = df.sort_values('time')
    df.reset_index(inplace=True)
    # Dynamically find the column containing timestamps ('time' in its name)
    time_column = "time"

    if not time_column:
        raise ValueError(
            "No column containing timestamps found. Please ensure a column with 'time' in its name exists.")

    # Convert the time column to pandas datetime format
    df[time_column] = pd.to_datetime(df[time_column])

    # Get the list of time series columns (excluding the time column)
    time_series_columns = ['AccX', 'AccY', 'AccZ', 'ThickIns']
    if filter:
        filt_y = butter_bandpass_filter(df[['AccX', 'AccY', 'AccZ']], lowcut=1, highcut=150, fs=1000)
        df.loc[:, ['AccX', 'AccY', 'AccZ']] = filt_y

    # Create a figure with len(time_series_columns) panels (subplots)
    fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(8, 4 * len(time_series_columns)))

    librosa.display.waveshow(np.float32(df['AccX']), sr=sample_rate, ax=ax[0])
    librosa.display.waveshow(np.float32(df['AccY']), sr=sample_rate, ax=ax[1])
    librosa.display.waveshow(np.float32(df['AccZ']), sr=sample_rate, ax=ax[2])
    ax[3].plot(np.float32(df['ThickIns']))

    mfccx = librosa.feature.mfcc(y=np.float32(df['AccX']), sr=sample_rate, n_fft=1024, hop_length=256,
                                 n_mfcc=8)
    mfccy = librosa.feature.mfcc(y=np.float32(df['AccY']), sr=sample_rate, n_fft=1024, hop_length=256,
                                 n_mfcc=8)
    mfccz = librosa.feature.mfcc(y=np.float32(df['AccZ']), sr=sample_rate, n_fft=1024, hop_length=256,
                                 n_mfcc=8)

    librosa.display.specshow(mfccx[2:, :], sr=sample_rate, ax=ax[4])
    librosa.display.specshow(mfccy[2:, :], sr=sample_rate, ax=ax[5])
    librosa.display.specshow(mfccz[2:, :], sr=sample_rate, ax=ax[6])

    # Set x-axis label using the detected time column
    plt.xlabel(time_column)

    # Add legends for each panel
    # for ax in ax:
    #     ax.legend()

    # Adjust layout for better readability
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_time_series_panels_with_median(df, window_size=3, hop_length=1):
    """
    Plot time series data in separate panels with windowed median using librosa.

    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data.
            The DataFrame should have a column containing timestamps with a name containing 'time',
            and one or more columns containing the time series data.

        window_size (int, optional): Size of the rolling window for calculating the median.
            Default is 3.

        hop_length (int, optional): Hop length for the rolling window.
            Default is 1, which means no overlap between windows.

    Returns:
        None (Plots the time series panels using matplotlib).
    """
    # Dynamically find the column containing timestamps ('time' in its name)
    time_column = [col for col in df.columns if 'time' in col.lower()]

    if not time_column:
        raise ValueError(
            "No column containing timestamps found. Please ensure a column with 'time' in its name exists.")

    time_column = time_column[0]

    # Convert the time column to pandas datetime format
    df[time_column] = pd.to_datetime(df[time_column])

    # Get the list of time series columns (excluding the time column)
    time_series_columns = df.columns.drop(time_column)

    # Create a figure with len(time_series_columns) panels (subplots)
    fig, axs = plt.subplots(nrows=len(time_series_columns), ncols=1, figsize=(8, 4 * len(time_series_columns)),
                            sharex=True)

    # Plot each time series in a separate panel
    for i, col in enumerate(time_series_columns):
        axs[i].plot(df[time_column], df[col], label=col)

        # Calculate the rolling median using librosa.sequence.rolling_window()
        x = df[col].values
        rolling_median = librosa.sequence.rolling_window(x, window_size, hop_length=hop_length, axis=0, pad=True,
                                                         dtype=None, **kwargs)
        rolling_median = pd.Series(rolling_median, index=df.index)
        axs[i].plot(df[time_column], rolling_median, label='Windowed Median', color='red')

        axs[i].set_ylabel(col)

    # Set x-axis label using the detected time column
    plt.xlabel(time_column)

    # Add legends for each panel
    for ax in axs:
        ax.legend()

    # Adjust layout for better readability
    plt.tight_layout()

    # Show the plot
    plt.show()
