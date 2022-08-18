import numpy as np
import pandas as pd
import itertools


from IPython.core.display import display, HTML

from devcode.extras import plot_eigenvalues
from devcode.models.lssvm import LSSVM
from devcode.utils import DataFrameUtils


def eigenvalues_logs(comb, freq, cond_worst, dtypes, hps_names):
    print("case: ", end='')

    _hps_values = [f"{hp_name} = {value:.2E}" for hp_name, value in zip(hps_names, comb)]
    _hps_str    = ";".join(_hps_values)

    print(f"Case: {_hps_str} [{str(freq)} instances")
    print(f"dtypes = {dtypes}")
    print(f"Worst conditioning: {cond_worst:.2E}\n\n\n")


def eigenvalues_analysis_local_and_regional(datasets, df_model, hps_names, eig_cond_func):
    for dataset_name in datasets:
        display(HTML('<center><h1>' + dataset_name + '</h1></center>'))

        # get dataframe of the specific dataset
        df_dataset = df_model.loc[df_model['dataset_name'] == dataset_name]

        hps_comb = np.unique(df_dataset[hps_names].values, axis=0)  # hyperparam combinations

        for comb in hps_comb:
            df_comb = DataFrameUtils.extract_ordered_case(df_dataset, hps_comb, hps_names)

            eigen_string = df_comb['eigenvalues'].values
            dtypes       = df_comb['eigenvalues_dtype'].values

            eigenvalues_list, cond_list = eig_cond_func(eigen_string, dtypes)

            freq = len(eigenvalues_list)

            # list of lists to single list
            merged = list(itertools.chain(*eigenvalues_list))

            eigenvalues = np.concatenate(merged)
            cond_worst = np.amax(cond_list)

            plot_eigenvalues(eigenvalues, dataset_name, comb, freq, show_flag=False)
            eigenvalues_logs(comb, freq, cond_worst, dtypes, hps_names)

        display(HTML('<hr>'))


def eigenvalues_analysis_local(datasets, df_model):
    hps_names = ["$k_{opt}$ [CV]", "$\gamma_{opt}$ [CV]", "$\sigma_{opt}$ [CV]"]
    return eigenvalues_analysis_local_and_regional(datasets, df_model, hps_names, local_eigenvalues_and_cond)


def eigenvalues_analysis_regional(datasets, df_model):
    hps_names = ["$k_{opt}$ [CV]", "$\gamma_{opt}$ [CV]", "$\sigma_{opt}$ [CV]"]
    return eigenvalues_analysis_local_and_regional(datasets, df_model, hps_names, local_eigenvalues_and_cond)


def eigenvalues_analysis_global(datasets, df_model):
    hps_names = ["$\gamma$", "$\sigma$"]
    return eigenvalues_analysis_local_and_regional(datasets, df_model, hps_names, eigenvalues_and_cond)


def eigenvalue_analysis(df_results):
    idx = 0
    eigvals_full = np.frombuffer(eval(df_results['eigenvalues'][idx]), dtype=df_results['eigenvalues_dtype'][idx])

    nan_indices  = np.argwhere(np.isnan(eigvals_full))
    eigvals_list = [None] * len(nan_indices)
    last_nan     = -1

    for i in range(len(nan_indices)):
        #     print(nan_indices[i][0])
        eigvals_list[i] = eigvals_full[last_nan + 1:nan_indices[i][0]]
        last_nan        = nan_indices[i][0]

    for eig in eigvals_list:
        print(eig)


def get_local_regional_eigenvaleus(local_models, X_local_labels, X_tr_norm):
    """
        Getting eigenvalues of Kernel matrices
    """
    n_models = len(local_models)
    # getting eigenvalues of kernel matrices
    n_lssvms = 0  # counter for LSSVM models
    models_idx = []
    for i in range(n_models):
        if isinstance(local_models[i], LSSVM):
            n_lssvms += 1
            models_idx.append(i)

    eigvals_list = [None, np.array([np.nan])] * n_lssvms
    count        = 0

    for i in models_idx:
        x_region = X_tr_norm[X_local_labels == i]
        K        = local_models[i].kernel(x_region, x_region)
        temp     = np.linalg.eigvals(K)

        eigvals_list[count] = temp
        count += 2

    eigvals = np.concatenate(eigvals_list, axis=0)

    return eigvals


def get_local_eigenvalues(local_based_model, X_tr_norm):
    return get_local_regional_eigenvaleus(local_based_model.models, local_based_model.ClusterAlg.labels_, X_tr_norm)


def get_regional_eigenvalues(local_based_model, X_tr_norm):
    X_local_labels = local_based_model.regionalize(X_tr_norm)
    return get_local_regional_eigenvaleus(local_based_model.regional_models, X_local_labels, X_tr_norm)


    # # getting eigenvalues of kernel matrices
    # n_lssvms   = 0  # counter for LSSVM models
    # models_idx = []
    # for i in range(len(local_model.models)):
    #     if isinstance(local_model.models[i], LSSVM):
    #         n_lssvms += 1
    #         models_idx.append(i)
    #
    # eigvals_list = [None, np.array([np.nan])] * n_lssvms
    # count        = 0
    # for i in models_idx:
    #     x_region = X_tr_norm[local_model.ClusterAlg.labels_ == i]
    #     K        = local_model.models[i].kernel(x_region, x_region)
    #     temp     = np.linalg.eigvals(K)
    #     eigvals_list[count] = temp  # .tostring()
    #     count += 2
    #
    # eigvals = np.concatenate(eigvals_list, axis=0)
    #
    # return eigvals


def eigenvalues_and_cond(eigen_string, dtypes):
    n_dtypes = len(dtypes)
    eigenvalues_list = [None] * n_dtypes
    cond_list        = np.empty(n_dtypes)  # conditioning

    for i in range(n_dtypes):
        eigenvalues_list[i] = np.frombuffer(eval(eigen_string[i]), dtype=dtypes[i])
        modules = np.absolute(eigenvalues_list[i])
        cond_list[i] = np.amax(modules) / np.amin(modules)

    return eigenvalues_list, cond_list


def local_eigenvalues_and_cond(eigen_string, dtypes):
    n_dtypes = len(dtypes)

    eigenvalues_list = [None] * n_dtypes
    cond_list        = []  # np.empty(len(dtypes)) # conditioning

    for i in range(n_dtypes):
        eigvals_full = np.frombuffer(eval(eigen_string[i]), dtype=dtypes[i])

        nan_indices  = np.argwhere(np.isnan(eigvals_full))
        eigvals_list = [None] * len(nan_indices)
        last_nan     = -1

        for j in range(len(nan_indices)):
            #     print(nan_indices[i][0])
            eigvals_list[j] = eigvals_full[last_nan + 1:nan_indices[j][0]]
            last_nan        = nan_indices[j][0]

            modules = np.absolute(eigvals_list[j])
            cond_list.append(np.amax(modules) / np.amin(modules))

        eigenvalues_list[i] = eigvals_list

    return eigenvalues_list, cond_list
