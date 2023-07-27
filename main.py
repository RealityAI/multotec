import os
import pandas as pd
import librosa
from helper import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

SAMPLE_RATE = 1000
FRAME_LEN = int(2 ** closest_power(SAMPLE_RATE * 0.2))
HOP_LEN = int(2 ** closest_power(SAMPLE_RATE * 0.1))
N_MFCC = 8


def target_extractor(feat_shape, target_col):
    _y = np.float32(np.array(target_col)).T
    out_y = librosa.feature.rms(y=_y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
    assert out_y.T.shape[0] == feat_shape[0]
    return out_y.T


def features_extractor(y):
    """
    y: amplitude data
    """

    # window = signal.windows.general_hamming(frame_len, 0.54, sym=True)
    mfccs = librosa.feature.mfcc(y=np.float32(np.array(y)).T, sr=SAMPLE_RATE, n_fft=FRAME_LEN, hop_length=HOP_LEN,
                                 n_mfcc=N_MFCC)
    mfccs = mfccs.reshape((mfccs.shape[0] * mfccs.shape[1], mfccs.shape[2]))
    return mfccs.T


def extract_features_row(file):
    file_name = os.path.join('data/', file)
    _y = pd.read_csv(file_name)
    feat = features_extractor(_y[['AccX', 'AccY']])
    target = target_extractor(feat_shape=feat.shape, target_col=_y['ThickIns'])
    return [file, feat, target]


def reshape_df(row):
    xx = pd.DataFrame(row['feature'])
    xx['target'] = row['target']
    xx['filename'] = row['filename']
    return xx


def load_features():
    audio_dataset_path = 'data/'
    files = os.listdir(audio_dataset_path)

    # Use Parallel to parallelize feature extraction
    extracted_features = Parallel(n_jobs=-1)(
        delayed(extract_features_row)(file) for file in tqdm(files)
    )

    extracted_features_df = pd.DataFrame(extracted_features,
                                         columns=['filename', 'feature', 'target'])

    # Use Parallel to parallelize DataFrame concatenation
    reshaped_dfs = Parallel(n_jobs=-1)(
        delayed(reshape_df)(row) for _, row in extracted_features_df.iterrows()
    )

    reshaped_df = pd.concat(reshaped_dfs, ignore_index=True)

    _X = np.array(reshaped_df.drop(["target", "filename"], axis=1))
    _y = np.array(reshaped_df[['target']])
    _groups = np.array(reshaped_df['filename'])

    return _X, _y, _groups


if __name__ == '__main__':
    # plot_phase_imbalance()
    # create_spectrogram()
    X, y, groups = load_features()
