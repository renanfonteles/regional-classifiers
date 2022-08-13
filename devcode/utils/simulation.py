from math import ceil

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

from devcode.analysis.clustering import organize_cluster_metrics, find_optimal_clusters
from devcode.analysis.eigenvalues import get_local_eigenvalues

from devcode.models.local_learning import LocalModel
from devcode.models.regional_learning import RegionalModel
from devcode.models.som import SOM
from devcode.models.lssvm import LSSVM

from devcode.utils import dummie2multilabel, collect_data
from devcode.utils.hyperparameters import search_lssvm_best_hp, search_local_lssvm_best_hp
from devcode.utils.metrics import DB


def _fit_and_evaluate_regional_model(case, X_tr_norm, y_train, X_ts_norm, y_test, hps_cases):
    # TODO: Old evalRLM
    """ Regional model """
    N = len(X_tr_norm)              # Number of datapoints in the train split

    som_params = case['som_params']
    l = ceil((5 * N ** .5) ** .5)   # side length of square grid of neurons

    som = SOM(l, l)

    C = l ** 2  # number of SOM neurons in the 2D grid
    k_values = [i for i in range(2, ceil(np.sqrt(C)))]  # 2 to sqrt(C)
    cluster_params = {
        'n_clusters': {'metric'  : DB,  # when a dictionary is pass a search begins
                       'criteria': np.argmin,  # search for smallest DB score
                       'k_values': k_values},  # around the values provided in 'k_values'
        'n_init': 10,  # number of initializations
        'init': 'random'
        # 'n_jobs':     0
    }

    linearModel = LinearRegression()

    rlm = RegionalModel(som, linearModel)
    rlm.fit(X=X_tr_norm, Y=y_train,
            SOM_params=som_params,
            Cluster_params=cluster_params)

    # Evaluating in the train set
    y_tr_pred = rlm.predict(X_tr_norm)
    y_tr_pred = np.round(np.clip(y_tr_pred, 0, 1))  # rounding prediction numbers

    cm_tr = confusion_matrix(dummie2multilabel(y_train),
                             dummie2multilabel(y_tr_pred)
                             ).reshape(-1)  # matrix => array

    # Evaluating in the test set
    y_ts_pred = rlm.predict(X_ts_norm)
    y_ts_pred = np.round(np.clip(y_ts_pred, 0, 1))  # rounding prediction numbers

    cm_ts = confusion_matrix(dummie2multilabel(y_test),
                             dummie2multilabel(y_ts_pred)
                             ).reshape(-1)  # matrix => array

    result_row = list(som_params.values()) + [cm_tr, cm_ts]

    return result_row


def _fit_and_evaluate_global_ols(case, X_tr_norm, y_train, X_ts_norm, y_test, hps_cases):
    # TODO: Old evalGOLS
    """ Global Ordinary Least Square (OLS) """
    model = LinearRegression().fit(X_tr_norm, y_train)

    # Evaluating in the train set
    y_tr_pred = model.predict(X_tr_norm)
    y_tr_pred = np.round(np.clip(y_tr_pred, 0, 1))  # rounding prediction numbers

    cm_tr = confusion_matrix(dummie2multilabel(y_train),
                             dummie2multilabel(y_tr_pred)
                             ).flatten()  # matrix => array

    # Evaluating in the test set
    y_ts_pred = model.predict(X_ts_norm)
    y_ts_pred = np.round(np.clip(y_ts_pred, 0, 1))  # rounding prediction numbers

    cm_ts = confusion_matrix(dummie2multilabel(y_test),
                             dummie2multilabel(y_ts_pred)
                             ).flatten()  # matrix => array

    result_row = [cm_tr, cm_ts]

    return result_row


def _fit_and_evaluate_global_lssvm(case, X_tr_norm, y_train, X_ts_norm, y_test, hps_cases):
    # TODO: Old eval_GLSSVM
    """ Global Least-Squares Support Vector Machine """
    best_hps = search_lssvm_best_hp(X_tr_norm, y_train, hps_cases)

    # Fit the model on best hyperparameters
    clf = LSSVM(kernel='rbf', **best_hps)
    clf.fit(X_tr_norm, y_train)

    # make predictions and evaluate model
    y_pred_tr, y_pred_ts = clf.predict(X_tr_norm), clf.predict(X_ts_norm)

    cm_tr = confusion_matrix(dummie2multilabel(y_train), dummie2multilabel(y_pred_tr))
    cm_ts = confusion_matrix(dummie2multilabel(y_test), dummie2multilabel(y_pred_ts))

    # Getting eigenvalues of kernel matrix
    K = clf.kernel(X_tr_norm, X_tr_norm)
    eig_vals = np.linalg.eigvals(K)

    result_row = [best_hps['gamma'], best_hps['sigma'], eig_vals.tostring(), eig_vals.dtype,
                  cm_tr.tostring(), cm_ts.tostring()]

    return result_row


def _fit_and_evaluate_local_lssvm(case, X_tr_norm, y_train, X_ts_norm, y_test, hps_cases):
    # TODO: Old eval_LLSSVM
    """ Local Least-Squares Support Vector Machine """

    # Fit the model on best hyperparameters
    best_hps, best_hps_list, best_k, suggestions, validation_scores = search_local_lssvm_best_hp(
        X_tr_norm, y_train, hps_cases)

    cluster_params = {'n_clusters': best_k, 'n_init': 10, 'init': 'random'}
    model_alg      = LSSVM(kernel='rbf', **best_hps)

    lm = LocalModel(ClusterAlg=KMeans, ModelAlg=model_alg)
    lm.fit(X_tr_norm, y_train, Cluster_params=cluster_params)

    y_pred_tr, y_pred_ts = lm.predict(X_tr_norm), lm.predict(X_ts_norm)
    cm_tr = confusion_matrix(dummie2multilabel(y_train), dummie2multilabel(y_pred_tr))
    cm_ts = confusion_matrix(dummie2multilabel(y_test), dummie2multilabel(y_pred_ts))

    empty_and_homo, valid_metrics = find_optimal_clusters(lm, suggestions, validation_scores, best_hps_list)

    eig_vals         = get_local_eigenvalues(lm, X_tr_norm)
    hyperparams      = [best_hps['gamma'], best_hps['sigma']]
    temp             = organize_cluster_metrics(best_hps_list, validation_scores, suggestions)
    suggestions_vals = [value for value in list(suggestions.values())]
    eig_metrics      = [eig_vals.tostring(), eig_vals.dtype, cm_tr.tostring(), cm_ts.tostring()]

    k_opt = len(lm.models)

    result_row = empty_and_homo + hyperparams + temp + [k_opt] + suggestions_vals + valid_metrics + eig_metrics

    # saving clusters to .csv
    # clusters = lm.ClusterAlg.cluster_centers_
    # clt_filename = "./temp_llssvm_cbic/clusters/L-LSSVM - {}.csv".format(case)
    # pd.DataFrame(clusters).to_csv(clt_filename, header=None, index=None)

    return result_row


def _save_results(result_row, filename, header):
    data = np.array(result_row).reshape(1, -1)

    results_df = pd.DataFrame(data, columns=header)
    results_df.to_csv(filename, mode='a', index=False, header=False)    # Append results to csv file


def run_simulation(datasets, header, case, scale_type, test_size, hps_cases, pipeline_func, simulation_file):
    dataset_name = case['dataset_name']
    random_state = case['random_state']

    X_tr_norm, y_train, X_ts_norm, y_test = collect_data(datasets, dataset_name, random_state, test_size, scale_type)

    result_row = pipeline_func(case, X_tr_norm, y_train, X_ts_norm, y_test, hps_cases)
    result_row = [dataset_name, random_state] + result_row

    _save_results(result_row, simulation_file.__str__(), header)


def eval_GLSSVM(datasets, simulation_file, header, scale_type, test_size, hps_cases, case):
    pipeline_func = _fit_and_evaluate_global_lssvm
    return run_simulation(datasets, header, case, scale_type, test_size, hps_cases, pipeline_func, simulation_file)


def eval_LLSSVM(datasets, simulation_file, header, scale_type, test_size, hps_cases, case):
    pipeline_func = _fit_and_evaluate_local_lssvm
    return run_simulation(datasets, header, case, scale_type, test_size, hps_cases, pipeline_func, simulation_file)


def evalGOLS(datasets, simulation_file, header, scale_type, test_size, hps_cases, case):
    pipeline_func = _fit_and_evaluate_global_ols
    return run_simulation(datasets, header, case, scale_type, test_size, hps_cases, pipeline_func, simulation_file)


def evalRLM(datasets, simulation_file, header, scale_type, test_size, hps_cases, case):
    pipeline_func = _fit_and_evaluate_regional_model
    return run_simulation(datasets, header, case, scale_type, test_size, hps_cases, pipeline_func, simulation_file)
