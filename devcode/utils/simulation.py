from math import ceil

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

from devcode.analysis.clustering import organize_cluster_metrics, find_optimal_clusters, get_k_opt_suggestions, \
    cluster_val_metrics, regional_cluster_val_metrics, RegionalUtils
from devcode.analysis.eigenvalues import get_local_eigenvalues, get_regional_eigenvalues

from devcode.models.local_learning import LocalModel
from devcode.models.regional_learning import RegionalModel
from devcode.models.som import SOM
from devcode.models.lssvm import LSSVM

from devcode.utils import dummie2multilabel, collect_data
from devcode.utils.hyperparameters import HyperOptimization
from devcode.utils.metrics import DB


def _evaluate_model(model, X, y, round_flag=False):
    # Evaluating in the test set
    y_ts_pred = model.predict(X)

    if round_flag:
        y_ts_pred = np.round(np.clip(y_ts_pred, 0, 1))  # rounding prediction numbers

    cm_ts = confusion_matrix(dummie2multilabel(y), dummie2multilabel(y_ts_pred)).reshape(-1)

    return cm_ts


def _evaluate_train_test_sets(X_tr_norm, y_train, X_ts_norm, y_test, model, round_flag=False):
    # Evaluating in the train set
    cm_tr = _evaluate_model(model, X=X_tr_norm, y=y_train, round_flag=round_flag)

    # Evaluating in the test set
    cm_ts = _evaluate_model(model, X=X_ts_norm, y=y_test, round_flag=round_flag)

    return cm_tr, cm_ts


def _fit_and_evaluate_regional_model(case, X_tr_norm, y_train, X_ts_norm, y_test, hps_cases):
    """ Regional model """
    dataset_name, random_state = case["dataset_name"], case["random_state"]

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
    }

    region_params = RegionalUtils.search_optimal_region_params("R-OLS", dataset_name, random_state, cluster_params,
                                                               som.neurons)

    linearModel = LinearRegression()

    rlm = RegionalModel(som, linearModel)
    rlm.fit(X=X_tr_norm, Y=y_train, SOM_params=som_params, cluster_params=region_params)

    cm_tr, cm_ts = _evaluate_train_test_sets(X_tr_norm, y_train, X_ts_norm, y_test, model=rlm, round_flag=True)

    result_row = list(som_params.values()) + [cm_tr, cm_ts]

    return result_row


def _fit_and_evaluate_global_ols(case, X_tr_norm, y_train, X_ts_norm, y_test, hps_cases):
    """ Global Ordinary Least Square (OLS) """
    ols = LinearRegression()
    ols.fit(X_tr_norm, y_train)

    cm_tr, cm_ts = _evaluate_train_test_sets(X_tr_norm, y_train, X_ts_norm, y_test, model=ols, round_flag=True)

    result_row = [cm_tr, cm_ts]

    return result_row


def _fit_and_evaluate_global_lssvm(case, X_tr_norm, y_train, X_ts_norm, y_test, hps_cases):
    """ Global Least-Squares Support Vector Machine """
    dataset_name, random_state = case["dataset_name"], case["random_state"]

    # Fit the model on best hyperparameters
    opt_params = HyperOptimization.search_best_params("G-LSSVM", HyperOptimization.search_lssvm_best_hp,
                                                      dataset_name, random_state, X_tr_norm, y_train, hps_cases)

    best_hps = opt_params

    # Fit the model on best hyperparameters
    clf = LSSVM(kernel='rbf', **best_hps)
    clf.fit(X_tr_norm, y_train)

    cm_tr, cm_ts = _evaluate_train_test_sets(X_tr_norm, y_train, X_ts_norm, y_test, model=clf, round_flag=False)

    # Getting eigenvalues of kernel matrix
    K = clf.kernel(X_tr_norm, X_tr_norm)
    eig_vals = np.linalg.eigvals(K)

    result_row = [best_hps['gamma'], best_hps['sigma'], eig_vals.tostring(), eig_vals.dtype,
                  cm_tr.tostring(), cm_ts.tostring()]

    return result_row


def _fit_and_evaluate_local_lssvm(case, X_tr_norm, y_train, X_ts_norm, y_test, hps_cases):
    """ Local Least-Squares Support Vector Machine """
    dataset_name, random_state = case["dataset_name"], case["random_state"]

    # Fit the model on best hyperparameters
    opt_params = HyperOptimization.search_best_params("L-LSSVM", HyperOptimization.search_local_lssvm_best_hp,
                                                      dataset_name, random_state, X_tr_norm, y_train, hps_cases)

    best_hps, best_hps_list, best_k, suggestions, validation_scores = opt_params

    cluster_params = {'n_clusters': best_k, 'n_init': 10, 'init': 'random'}
    model_alg      = LSSVM(kernel='rbf', **best_hps)

    lm = LocalModel(ClusterAlg=KMeans, ModelAlg=model_alg)
    lm.fit(X_tr_norm, y_train, Cluster_params=cluster_params)

    cm_tr, cm_ts = _evaluate_train_test_sets(X_tr_norm, y_train, X_ts_norm, y_test, model=lm, round_flag=False)

    n_empty_regions = len(lm.empty_regions)
    empty_and_homo, valid_metrics = find_optimal_clusters(lm.models, n_empty_regions, suggestions, validation_scores,
                                                          best_hps_list, cluster_val_metrics)

    eig_vals         = get_local_eigenvalues(lm, X_tr_norm)
    hyperparams      = [best_hps['gamma'], best_hps['sigma']]
    temp             = organize_cluster_metrics(best_hps_list, validation_scores, suggestions,
                                                cluster_val_metrics)        # optimal_k_hps
    suggestions_vals = [value for value in list(suggestions.values())]
    eig_metrics      = [eig_vals.tostring(), eig_vals.dtype, cm_tr.tostring(), cm_ts.tostring()]

    k_opt = len(lm.models)

    result_row = empty_and_homo + hyperparams + temp + [k_opt] + suggestions_vals + valid_metrics + eig_metrics

    # saving clusters to .csv
    # clusters = lm.ClusterAlg.cluster_centers_
    # clt_filename = f"./temp_llssvm_cbic/clusters/L-LSSVM - {case}.csv"
    # pd.DataFrame(clusters).to_csv(clt_filename, header=None, index=None)

    return result_row


def _fit_and_evaluate_regional_lssvm(case, X_tr_norm, y_train, X_ts_norm, y_test, hps_cases):
    """ Regional Least-Squares Support Vector Machine """
    dataset_name, random_state = case["dataset_name"], case["random_state"]

    # Fit the model on best hyperparameters
    opt_params = HyperOptimization.search_best_params("R-LSSVM", HyperOptimization.search_regional_lssvm_best_hp,
                                                      dataset_name, random_state, X_tr_norm, y_train, hps_cases)

    best_hps, best_hps_list, best_k, suggestions, validation_scores, som_tr = opt_params

    cluster_params = {'n_clusters': best_k, 'n_init': 10, 'init': 'random'}
    region_params  = RegionalUtils.search_optimal_region_params("R-LSSVM", dataset_name, random_state, cluster_params,
                                                                som_tr.neurons)

    model_alg      = LSSVM(kernel='rbf', **best_hps)

    rm = RegionalModel(som_tr, model_alg)
    rm.fit(X=X_tr_norm, Y=y_train, verboses=0, cluster_params=region_params)

    cm_tr, cm_ts = _evaluate_train_test_sets(X_tr_norm, y_train, X_ts_norm, y_test, model=rm, round_flag=False)

    n_empty_regions = len(rm.empty_regions)
    empty_and_homo, valid_metrics = find_optimal_clusters(
        rm.regional_models, n_empty_regions, suggestions, validation_scores, best_hps_list,
        regional_cluster_val_metrics)

    eig_vals         = get_regional_eigenvalues(rm, X_tr_norm)
    hyperparams      = [best_hps['gamma'], best_hps['sigma']]
    temp             = organize_cluster_metrics(best_hps_list, validation_scores, suggestions,
                                                regional_cluster_val_metrics)
    suggestions_vals = [value for value in list(suggestions.values())]
    eig_metrics      = [eig_vals.tostring(), eig_vals.dtype, cm_tr.tostring(), cm_ts.tostring()]

    k_opt = len(rm.regional_models)

    result_row = empty_and_homo + hyperparams + temp + [k_opt] + suggestions_vals + valid_metrics + eig_metrics

    # saving clusters to .csv
    # kmeans_proto = rm.Cluster.cluster_centers_
    # neurons = rm.SOM.neurons
    #
    # kmeans_filename  = (f"./temp_rlssvm/clusters/kmeans/L-LSSVM - {case}.csv").replace(':', '-')
    # neurons_filename = (f"./temp_rlssvm/clusters/som/L-LSSVM - {case}.csv").replace(':', '-')
    #
    # pd.DataFrame(kmeans_proto).to_csv(kmeans_filename, header=None, index=None)
    # pd.DataFrame(neurons).to_csv(neurons_filename, header=None, index=None)

    return result_row


def _save_results(result_row, filename, header):
    data = np.array(result_row).reshape(1, -1)

    results_df = pd.DataFrame(data, columns=header)
    results_df.to_csv(filename, mode='a', index=False, header=False)    # Append results to csv file


def set_per_round(cases, datasets, test_size, scale_type):
    return [collect_data(datasets, case['dataset_name'], case['random_state'], test_size, scale_type)
            for case in cases]


def run_simulation(model_name, datasets, header, case, scale_type, test_size, hps_cases, pipeline_func,
                   simulation_file):
    dataset_name = case['dataset_name']
    random_state = case['random_state']

    X_tr_norm, y_train, X_ts_norm, y_test = collect_data(datasets, dataset_name, random_state, test_size, scale_type)

    result_row = pipeline_func(case, X_tr_norm, y_train, X_ts_norm, y_test, hps_cases)
    result_row = [dataset_name, random_state] + result_row

    _save_results(result_row, simulation_file.__str__(), header)


def eval_GLSSVM(datasets, simulation_file, header, scale_type, test_size, hps_cases, case):
    pipeline_func = _fit_and_evaluate_global_lssvm

    return run_simulation("G-LSSVM", datasets, header, case, scale_type, test_size, hps_cases, pipeline_func,
                          simulation_file)


def eval_LLSSVM(datasets, simulation_file, header, scale_type, test_size, hps_cases, case):
    pipeline_func = _fit_and_evaluate_local_lssvm

    return run_simulation("L-LSSVM", datasets, header, case, scale_type, test_size, hps_cases, pipeline_func,
                          simulation_file)


def evalGOLS(datasets, simulation_file, header, scale_type, test_size, hps_cases, case):
    pipeline_func = _fit_and_evaluate_global_ols

    return run_simulation("G-OLS", datasets, header, case, scale_type, test_size, hps_cases, pipeline_func,
                          simulation_file)


def evalRLM(datasets, simulation_file, header, scale_type, test_size, hps_cases, case):
    pipeline_func = _fit_and_evaluate_regional_model

    return run_simulation("R-OLS", datasets, header, case, scale_type, test_size, hps_cases, pipeline_func,
                          simulation_file)


def eval_RLSSVM(datasets, simulation_file, header, scale_type, test_size, hps_cases, case):
    pipeline_func = _fit_and_evaluate_regional_lssvm

    return run_simulation("R-LSSVM", datasets, header, case, scale_type, test_size, hps_cases, pipeline_func,
                          simulation_file)
