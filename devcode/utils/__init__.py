import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataFrameUtils:
    @staticmethod
    def extract_ordered_case(df, df_params, s_params):
        n_params = len(s_params)
        _filter_params = [(df[df_params[i]] == s_params[i]) for i in range(n_params)]

        combined_filter = _filter_params[0]

        for _filter in _filter_params:
            combined_filter = combined_filter & _filter

        df_case = df.loc[combined_filter]

        return df_case


def data_preprocessing(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    temp   = scaler.fit_transform(data.data)

    pca = PCA(n_components=2)
    X   = pca.fit_transform(temp)

    return X


def process_labels(y_train):
    # solving multilabel problem in wall-following data set
    y_temp = y_train
    if y_train.ndim == 2:
        if y_train.shape[1] >= 2:
            y_temp = dummie2multilabel(y_train)

    return y_temp


def collect_data(datasets, dataset_name, random_state, test_size, scale_type):
    X = datasets[dataset_name]['features'].values
    Y = datasets[dataset_name]['labels'].values

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state,
                                                        stratify=np.unique(Y, axis=1))
    # scaling features
    X_tr_norm, X_ts_norm = scale_feat(X_train, X_test, scaleType=scale_type)

    return X_tr_norm, y_train, X_ts_norm, y_test


def scale_feat(X_train, X_test, scaleType='min-max'):
    """ Function to scale data """

    if scaleType == 'min-max' or scaleType == 'std':
        X_tr_norm = np.copy(X_train)    # making a copy to leave original available
        X_ts_norm = np.copy(X_test)
        scaler = MinMaxScaler() if scaleType == 'min-max' else StandardScaler()
        scaler.fit(X_tr_norm)
        X_tr_norm = scaler.transform(X_tr_norm)
        X_ts_norm = scaler.transform(X_ts_norm)
        return (X_tr_norm, X_ts_norm)
    else:
        raise ValueError("Scale type not defined. Use 'min-max' or 'std' instead.")


def dummie2multilabel(X):
    """ Convert dummies to multilable """
    N = len(X)
    X_multi = np.zeros((N, 1), dtype='int')
    for i in range(N):
        temp = np.where(X[i] == 1)[0]   # Find where 1 is found in the array
        if temp.size == 0:              # is an empty array, there is no '1' in the X[i] array
            X_multi[i] = 0              # so we denote this class '0'
        else:
            X_multi[i] = temp[0] + 1    # we have +1 because 0 denote the class with an empty array
    return X_multi.T[0]


def cm2acc(cm):
    """ Takes confusion matrix and evaluate the accuracy """
    acc = 0
    total = sum(sum(cm))

    for j in range(cm.shape[0]):
        acc += cm[j, j]         # Summing the diagonal

    acc /= total
    return acc


def is_over():
    """ Auxiliary function to tell when processing is over (Linux OS) """
    import os
    os.system('spd-say "your program has finished"')


def printDateTime():
    print(datetime.datetime.now())


def load_csv_as_pandas(path, sort_flag=False):
    import glob
    import pandas as pd

    all_files = glob.glob(path + "/*.csv")

    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df_results = pd.concat(li, axis=0, ignore_index=True)

    if sort_flag:
        df_results.sort_values(by='dataset_name')

    return df_results


def initialize_file(filename, header):
    from pathlib import Path
    simulation_file = Path(filename)

    import csv

    # open the file in the write mode
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    return simulation_file
