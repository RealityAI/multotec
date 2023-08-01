import os
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import *


def extract_raw_data(file):
    file_name = os.path.join('data/', file)
    _y = pd.read_csv(file_name)
    panel_number = int(file.split('_')[1])
    subset = _y[['DateTime', 'AccX', 'AccY', 'AccZ', 'ThickIns']]
    subset = subset.copy()
    subset.loc[:, 'panel_number'] = panel_number
    subset.loc[:, 'file'] = file
    subset.loc[:, 'time'] = generate_time_sequence(subset['DateTime'].unique()[0], subset.shape[0])
    subset.reset_index(inplace=True)
    return subset


def load_raw_data():
    audio_dataset_path = 'data/'
    files = os.listdir(audio_dataset_path)

    # Use Parallel to parallelize feature extraction
    extracted_data = Parallel(n_jobs=CORES)(
        delayed(extract_raw_data)(file) for file in tqdm(files)
    )

    extracted_data_df = pd.concat(extracted_data)
    plot_time_series_for_one_panel(extracted_data_df)
    print("check")
    return extracted_data_df


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
    _X = np.array(reshaped_df.drop(["target", "filename", "panel_number"], axis=1))
    _y = np.array(reshaped_df[['target']])
    _y = [ix[0] for ix in _y]
    _groups = np.array(reshaped_df['panel_number'])
    return _X, _y, _groups

def build_regression_model(_X,_y,_groups):


if __name__ == '__main__':
    load_raw_data()
    # plot_phase_imbalance()
    # create_spectrogram()
    X, y, groups = load_features()
    # view_features()
    # y = np.array([x[0] for x in y])
