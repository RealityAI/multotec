import os
import pandas as pd
import librosa
from helper import *
import matplotlib.pyplot as plt


def features_extractor(y):
    """
    y: amplitude data
    """
    sample_rate = 1000
    frame_len = int(2 ** closest_power(sample_rate / 1000 * 10240))
    hop_len = int(2 ** closest_power(sample_rate / 1000 * 1024))
    # window = signal.windows.general_hamming(frame_len, 0.54, sym=True) mfccs = librosa.feature.mfcc(y=np.float32(
    # np.array(y)).T, sr=sample_rate, n_fft=frame_len, hop_length=hop_len, n_mfcc=8, window=window)
    mfccs = librosa.feature.mfcc(y=np.float32(np.array(y)).T, sr=sample_rate, n_fft=frame_len, hop_length=hop_len,
                                 n_mfcc=8)
    mfccs = mfccs.reshape((mfccs.shape[0] * mfccs.shape[1], mfccs.shape[2]))
    return mfccs.T


def load_features():
    audio_dataset_path = 'data/'
    files = os.listdir(audio_dataset_path)
    extracted_features = []
    for file in files:
        print(file)
        file_name = os.path.join('data/', file)
        _y = pd.read_csv(file_name)
        feat = features_extractor3(_y[['Iu', 'Iv']])


if __name__ == '__main__':
    # plot_phase_imbalance()
    # create_spectrogram()
    X, y, groups = load_features()
