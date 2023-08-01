from scipy import stats
from utils.helper import *
import os


SAMPLE_RATE = 1000
N_FFT = FRAME_LEN = int(2 ** closest_power(SAMPLE_RATE * 0.2))
HOP_LEN = int(2 ** closest_power(SAMPLE_RATE * 0.1))
N_MELS = N_MFCC = 8
CORES = 2


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


def generate_time_sequence(start_time, num_samples, sampling_rate_hz=1000):
    """
    Generate a sequence of time stamps at a given sampling rate.

    Parameters:
        start_time (str or pd.Timestamp): The starting time value.
        num_samples (int): Number of time stamps to generate.
        sampling_rate_hz (int, optional): Sampling rate in Hz (samples per second).
            Default is 1000.

    Returns:
        pd.Series: A Series containing the generated time stamps.
    """
    # Convert the start_time to pandas Timestamp if it's not already in that format
    if not isinstance(start_time, pd.Timestamp):
        start_time = pd.Timestamp(start_time)

    # Calculate the time difference between consecutive time stamps
    time_diff = pd.Timedelta(seconds=1 / sampling_rate_hz)

    # Generate the sequence of time stamps using NumPy
    time_stamps = np.arange(start_time, start_time + num_samples * time_diff, time_diff)

    return pd.Series(time_stamps)