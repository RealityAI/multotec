import os
from scipy import signal
import pandas as pd
import librosa
from helper import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import random
from scipy import stats

SAMPLE_RATE = 1000
N_FFT = FRAME_LEN = int(2 ** closest_power(SAMPLE_RATE * 0.2))
HOP_LEN = int(2 ** closest_power(SAMPLE_RATE * 0.1))
N_MELS = N_MFCC = 8
CORES = 2


def plot_target_with_panel_changes(df):
    # Convert the 'time' column to pandas datetime format
    df['time'] = pd.to_datetime(df['time'])

    # Discretize the 'panel_number' column and convert it to a categorical variable
    df['panel_number'] = pd.Categorical(df['panel_number'])

    # Get unique panel numbers
    unique_panels = df['panel_number'].cat.codes.unique()

    # Create a time series plot for each panel number
    for panel_code in unique_panels:
        panel_df = df[df['panel_number'].cat.codes == panel_code]
        plt.plot(panel_df['time'], panel_df['target'], label=f'Panel {panel_code}', marker = 'o',alpha=0.3)

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


def target_extractor(feat_shape, target_col):
    _y = np.float32(np.array(target_col)).T
    return np.repeat(stats.mode(_y, axis=0, keepdims=False)[0], feat_shape[0])


def features_extractor(_y):
    """
    y: amplitude data
    """

    # window = signal.windows.general_hamming(frame_len, 0.54, sym=True)
    mfccs = librosa.feature.mfcc(y=np.float32(np.array(_y)).T, sr=SAMPLE_RATE, n_fft=FRAME_LEN, hop_length=HOP_LEN,
                                 n_mfcc=N_MFCC, center=False)
    mfccs = mfccs.reshape((mfccs.shape[0] * mfccs.shape[1], mfccs.shape[2]))
    return mfccs.T


def extract_features_row(file):
    file_name = os.path.join('data/', file)
    _y = pd.read_csv(file_name)
    feat = features_extractor(_y[['AccX', 'AccY']])
    target = target_extractor(feat_shape=feat.shape, target_col=_y['ThickIns'])
    panel_number = int(file.split('_')[1])
    # time = pd.to_datetime(_y['DateTime'], format='%d-%b-%Y %H:%M:%S')
    time = _y['DateTime'][0]
    return [time, file, feat, target, panel_number]


def reshape_df(row):
    xx = pd.DataFrame(row['feature'])
    xx['target'] = row['target']
    xx['filename'] = row['filename']
    xx['panel_number'] = row['panel_number']
    xx['time'] = row['time']
    return xx


def load_features():
    audio_dataset_path = 'data/'
    files = os.listdir(audio_dataset_path)

    # Use Parallel to parallelize feature extraction
    extracted_features = Parallel(n_jobs=CORES)(
        delayed(extract_features_row)(file) for file in tqdm(files)
    )

    extracted_features_df = pd.DataFrame(extracted_features,
                                         columns=['time', 'filename', 'feature', 'target', 'panel_number'])

    # Use Parallel to parallelize DataFrame concatenation
    reshaped_dfs = Parallel(n_jobs=CORES)(
        delayed(reshape_df)(row) for _, row in extracted_features_df.iterrows()
    )

    reshaped_df = pd.concat(reshaped_dfs, ignore_index=True)
    reshaped_df['time'] = pd.to_datetime(reshaped_df['time'], format='%d-%b-%Y %H:%M:%S')

    plot_target_with_panel_changes(reshaped_df)
    _X = np.array(reshaped_df.drop(["target", "filename"], axis=1))
    _y = np.array(reshaped_df[['target']])
    _groups = np.array(reshaped_df['panel_number'])
    sub = reshaped_df[reshaped_df['panel_number'] == 101]
    sub = sub.sort_values(by=['time'])
    return _X, _y, _groups


#
# def view_features(hop_len=HOP_LEN, frame_len=FRAME_LEN, sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, n_files=4):
#     audio_dataset_path = 'data/'
#     files = random.sample(os.listdir(audio_dataset_path), n_files)
#     fig, ax = plt.subplots(nrows=6, ncols=n_files, figsize=(40, 40))
#     for i, file in enumerate(files):
#         filename = audio_dataset_path + file
#         y = pd.read_csv(filename)
#         y = np.float32(np.array(y)).T
#         category = i
#         # window = signal.windows.general_hamming(frame_len, 0.54, sym=True)
#         window = signal.windows.boxcar(FRAME_LEN)
#         # get log- frequency power spectrogram
#         D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_len)),
#                                     ref=np.max)
#
#         # get the melspec in db before plotting
#         melspec = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=frame_len, hop_length=hop_len,
#                                                  n_mels=n_mels)
#         melspec = librosa.power_to_db(melspec, ref=np.max)
#
#         # get the mfccs
#         mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_fft=frame_len, hop_length=hop_len, window=window,
#                                     n_mfcc=N_MFCC)
#         # get the tempogram
#         oenv = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_len)
#         tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_len, norm=None)
#         # get the chromagram
#         chromagram = librosa.feature.chroma_stft(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_len, n_chroma=N_MELS)
#
#         # plot the wave
#         librosa.display.waveshow(y, sr=sample_rate, ax=ax[0][i])
#         # plot the mfcc
#         librosa.display.specshow(mfcc, sr=sample_rate, ax=ax[1][i], y_axis='mel', x_axis='time')
#         # plot the melspec
#         librosa.display.specshow(melspec, sr=sample_rate, ax=ax[2][i], y_axis='mel', x_axis='time')
#         # plot the tempogram
#         librosa.display.specshow(tempogram, sr=sample_rate, hop_length=hop_len, x_axis='time', y_axis='tempo',
#                                  ax=ax[3][i])
#         # plot the chromagram
#         librosa.display.specshow(chromagram, sr=sample_rate, hop_length=hop_len, x_axis='time', y_axis='chroma',
#                                  ax=ax[4][i])
#         # plot the power spectrogram
#         librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_len, fmax=sr / 2,
#                                  x_axis='time', ax=ax[5][i])
#
#         ax[0][i].set_title(f"Raw audio signal | Class: {category} | Shape: {_y.shape}", fontsize=20)
#         ax[1][i].set_title(f"MFCC | Class: {category} | Shape: {mfcc.T.shape}", fontsize=20)
#         ax[2][i].set_title(f"Mel Spec | Class: {category} | Shape: {melspec.T.shape}", fontsize=20)
#         ax[3][i].set_title(f"Tempogram | Class: {category} | Shape: {tempogram.T.shape}", fontsize=20)
#         ax[4][i].set_title(f"Chromogram | Class: {category} | Shape: {chromagram.T.shape}", fontsize=20)
#         ax[5][i].set_title(f"Power Spectrogram | Class: {category} | Shape: {D.T.shape}", fontsize=20)
#
#     fig.suptitle('4 different examples, 5 different representations\n\n', fontsize=40)
#     fig.tight_layout()
#     fig.savefig('output/4_examples.png')
#     fig.show()


if __name__ == '__main__':
    # plot_phase_imbalance()
    # create_spectrogram()
    X, y, groups = load_features()
    print("YAY")
    # view_features()
    y = np.array([x[0] for x in y])

    num_bins = 20  # Number of bins for the histogram
    plt.hist([y[groups == group] for group in np.unique(groups)], bins=num_bins, alpha=0.7, label=np.unique(groups))
    plt.hist([y[groups == group] for group in [107]], bins=num_bins, alpha=0.7, label=np.unique(groups))

    # y_plot = y[y < 14]
    #
    # plt.hist(y_plot, density=True, bins=1000)
    # plt.show()
    # y_plot = y[(y > 15.4) & (y < 16)]
    # plt.hist(y_plot, density=True, bins=1000)
    # plt.show()
    # y_plot = y[y > 18.75]
    # plt.hist(y_plot, density=True, bins=1000)
    # plt.show()
