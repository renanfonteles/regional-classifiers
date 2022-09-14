import pandas as pd
from devcode.utils import scale_feat

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

paper_result_path = "results/paper-results"

datasets = None

def get_datasets(base_path="data"):
    vc2c_path = f"{base_path}/vertebral_column/column_2C.dat"
    vc3c_path = f"{base_path}/vertebral_column/column_3C.dat"

    wf24f_path = f"{base_path}/wall_following/sensor_readings_24.data"
    wf4f_path  = f"{base_path}/wall_following/sensor_readings_4.data"
    wf2f_path  = f"{base_path}/wall_following/sensor_readings_2.data"

    pk_path  = f"{base_path}/parkinson/parkinsons.data"

    # Vertebral Column

    # dataset for classification between Normal (NO) and Abnormal (AB)
    vc2c = pd.read_csv(vc2c_path, delim_whitespace=True, header=None)

    # dataset for classification between DH (Disk Hernia), Spondylolisthesis (SL) and Normal (NO)
    vc3c = pd.read_csv(vc3c_path, delim_whitespace=True, header=None)

    # Wall-Following
    # dataset with all 24 ultrassound sensors readings
    wf24f = pd.read_csv(wf24f_path, header=None)
    # dataset with simplified 4 readings (front, left, right and back)
    wf4f  = pd.read_csv(wf4f_path, header=None)
    # dataset with simplified 2 readings (front and left)
    wf2f  = pd.read_csv(wf2f_path, header=None)

    # Parkinson (31 people, 23 with Parkinson's disease (PD))
    temp       = pd.read_csv(pk_path)
    labels     = temp.columns.values.tolist()
    new_labels = [label for label in labels if label not in ('name')] # taking off column 'name'
    pk         = temp[new_labels]


    pk_features = pk.columns.tolist()
    pk_features.remove('status')

    # datasets with separation between 'features' and 'labels'
    datasets = {
        "vc2c":  {"features": vc2c.iloc[:, 0:6],        "labels": pd.get_dummies(vc2c.iloc[:, 6],   drop_first=True)},
        "vc3c":  {"features": vc3c.iloc[:, 0:6],        "labels": pd.get_dummies(vc3c.iloc[:, 6],   drop_first=True)},
        "wf24f": {"features": wf24f.iloc[:, 0:24],      "labels": pd.get_dummies(wf24f.iloc[:, 24], drop_first=True)},
        "wf4f":  {"features": wf4f.iloc[:, 0:4],        "labels": pd.get_dummies(wf4f.iloc[:, 4],   drop_first=True)},
        "wf2f":  {"features": wf2f.iloc[:, 0:2],        "labels": pd.get_dummies(wf2f.iloc[:, 2],   drop_first=True)},
        "pk":    {"features": pk.loc[:, pk_features],   "labels": pk.loc[:, ["status"]]}
    }

    return datasets

'''
OBS: Was chosen to maintain k-1 dummies variables when we had k categories, so the missing category is 
identified when all dummies variables are zero.
'''

import numpy as np

# printing datasets info

# print("{:10}{:18}{:}".format(
#         'Dataset:',
#         'Features.shape:',
#         '# of classes:',
#         ))
# for dataset_name, data in datasets.items():
#     print("{:9} {:17} {:}".format(
#         dataset_name,
#         str(data['features'].shape),
#         len(np.unique(data['labels'].values, axis=0))
#         ))


def print_available_datasets(base_path):
    datasets = get_datasets(base_path)
    for dataset in datasets:
        print("Dataset name: {}\nNumber of features: {}\nNumber of samples: {}\n".format(
            dataset, datasets[dataset]["features"].shape[1], datasets[dataset]["features"].shape[0]
        ))


def select_dataset(ds_name, base_path):
    datasets = get_datasets(base_path)

    if ds_name.lower() == "mnist":
        from sklearn.datasets import load_digits

        title      = 'MNIST data set after PCA (2 components)'
        datapoints = load_digits()
        scaler     = MinMaxScaler()
        temp       = scaler.fit_transform(datapoints.data)

        pca = PCA(n_components=2)
        X   = pca.fit_transform(temp)
    else:
        datapoints = datasets[ds_name]
        title      = f"{ds_name} dataset"
        X          = datapoints['features'].values.copy()
        X, _       = scale_feat(X, X, scaleType='min-max')

    return X, title

