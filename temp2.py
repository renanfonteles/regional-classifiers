import pandas as pd
import numpy as np
import plotly.offline as py

from devcode.analysis.clustering import local_cluster_val_metrics, regional_cluster_val_metrics
from devcode.analysis.results import results_per_dataset
from devcode.models.som import SOM
from devcode.simulation import ResultProcessor
from devcode.utils import load_csv_as_pandas, collect_data

from load_dataset import get_datasets

datasets = get_datasets()

# _dataframes_dict = {
#     'global'    : load_csv_as_pandas(path="results/local-results/cbic/temp_glssvm_cbic"),
#     'local'     : load_csv_as_pandas(path="results/local-results/cbic/temp_llssvm_cbic/results"),
#     'regional'  : load_csv_as_pandas(path="results/regional-results/temp_rlssvm_somfix/results")
# }
#

#
# py.init_notebook_mode(connected=True)  # enabling plot within jupyter notebook
#
# set_dict = {'treino': 'cm_tr', 'teste': 'cm_ts'}
#
# model_labels = _dataframes_dict.keys()


# ResultProcessor.process_results_in_multiple_datasets(ds_names, _dataframes_dict,
#                                                      ResultProcessor.regional_k_optimal_histogram)

import re
import os

from devcode.utils.hyperparameters import HyperOptimization

ds_names = ['pk', 'vc2c', 'vc3c', 'wf2f', 'wf4f', 'wf24f']


def get_file_names_in_dir(dir_path):
    return [path for path in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, path))]


def get_random_states(file_names):
    all_rnd_state = [int(re.split(" ", file_name)[-1][:-5]) for file_name in file_names]
    return list(set(all_rnd_state))


def cluster_metric_based_hps(df_round, header, cluster_metrics, cast=float):
    return [cast(df_round[f"{header} [{valid_name['name']}]"]) for valid_name in cluster_metrics]


def single_round_glssvm_hps(df_round, ds_name, random_state):
    return {"sigma": float(df_round["$\sigma$"]), "gamma": float(df_round["$\gamma$"])}


def single_round_local_based_lssvm_hps(df_round, cluster_metrics):
    best_hps = {"gamma": float(df_round['$\gamma_{opt}$ [CV]']), "sigma": float(df_round['$\sigma_{opt}$ [CV]'])}
    best_k   = int(df_round['$k_{opt}$ [CV]'])

    gammas = cluster_metric_based_hps(df_round, header="$\gamma_{opt}$", cluster_metrics=cluster_metrics)
    sigmas = cluster_metric_based_hps(df_round, header="$\sigma_{opt}$", cluster_metrics=cluster_metrics)

    best_hps_list = [{"gamma": gamma, "sigma": sigma} for gamma, sigma in zip(gammas, sigmas)]

    k_opt_by_metric = cluster_metric_based_hps(df_round, header="$k_{opt}$",
                                               cluster_metrics=cluster_metrics, cast=int)

    valid_metric_names = [valid_metric["name"] for valid_metric in cluster_metrics]
    suggestions        = dict(zip(valid_metric_names, k_opt_by_metric))

    valid_score_values = cluster_metric_based_hps(df_round, header="cv_score",
                                                  cluster_metrics=cluster_metrics)

    valid_scores_per_k = list(set([(_k, _score) for _k, _score in zip(k_opt_by_metric, valid_score_values)]))
    validation_scores  = np.array(valid_scores_per_k)
    validation_scores  = validation_scores[validation_scores[:, 0].argsort()]

    return best_hps, best_hps_list, best_k, suggestions, validation_scores


def single_round_local_lssvm_hps(df_round, ds_name, random_state):
    return single_round_local_based_lssvm_hps(df_round, cluster_metrics=local_cluster_val_metrics)


def single_round_regional_lssvm_hps(df_round, ds_name, random_state):
    som_params = {'alpha0': 0.1, 'sigma0': 10, 'nEpochs': 300, 'verboses': 0}

    X_tr_norm, y_train, X_ts_norm, y_test = collect_data(datasets, ds_name, random_state,
                                                         test_size=0.5, scale_type="min-max")

    # Fit the SOM in the whole train set so we can get suggestion for number of clusters in K-Means
    N = len(X_tr_norm)
    l = int((5 * N ** .5) ** .5)  # size of square grid of neurons
    som_tr = SOM(l, l)
    som_tr.fit(X_tr_norm, **som_params)

    best_hps, best_hps_list, best_k, suggestions, validation_scores = single_round_local_based_lssvm_hps(
        df_round, cluster_metrics=regional_cluster_val_metrics)

    return best_hps, best_hps_list, best_k, suggestions, validation_scores, som_tr


def extract_hps(df, rnd_states, extract_func, model_name):
    # hps_per_dataset = []
    for ds_name in ds_names:
        df_dataset  = df[df["dataset_name"] == ds_name]

        for random_state in rnd_states:
            df_round     = df_dataset[df_dataset["random_state"] == random_state]
            df_round_hps = extract_func(df_round, ds_name, random_state)

            HyperOptimization.save_optimal_hyperparams(df_round_hps, ds_name, random_state, model_name)

        # df_rounds   = [df_dataset[df_dataset["random_state"] == random_state] for random_state in rnd_states]
        # dataset_hps = [extract_func(df_round) for df_round in df_rounds]
        #
        # hps_per_dataset.append(dataset_hps)


    #     # TODO: Save .pickle files
    #
    # return hps_per_dataset


glssvm_result_path = "results/local-results/cbic/temp_glssvm_cbic"
llssvm_result_path = "results/local-results/cbic/temp_llssvm_cbic/results"
rlssvm_result_path = "results/regional-results/temp_rlssvm_somfix/results"

random_states = get_random_states(get_file_names_in_dir(dir_path=glssvm_result_path))

# 1. G-LSSVM
df_global  = load_csv_as_pandas(glssvm_result_path)
extract_hps(df_global, random_states, extract_func=single_round_glssvm_hps, model_name="G-LSSVM")

# 2. L-LSSVM
df_local        = load_csv_as_pandas(llssvm_result_path)
extract_hps(df_local, random_states, extract_func=single_round_local_lssvm_hps, model_name="L-LSSVM")

# 3. R-LSSVM
df_regional         = load_csv_as_pandas(rlssvm_result_path)
extract_hps(df_regional, random_states, extract_func=single_round_regional_lssvm_hps, model_name="R-LSSVM")

print("Final")
