import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from devcode import PRE_PROCESS_PATH
from devcode.analysis.clustering import get_k_opt_suggestions, cluster_val_metrics, find_optimal_clusters, \
    regional_cluster_val_metrics

from devcode.models.local_learning import LocalModel
from devcode.models.lssvm import LSSVM
from devcode.models.regional_learning import RegionalModel
from devcode.models.som import SOM
from devcode.utils import process_labels, FileUtils
from devcode.utils.evaluation import f_o


default_som_params = {'alpha0': 0.1, 'sigma0': 10, 'nEpochs': 300, 'verboses': 0}


def _train_regional_model(x_tr, y_tr, cluster_params, model_alg, som_params):
    # Train the model on the train set
    N = len(x_tr)
    l = int((5 * N ** .5) ** .5)  # size of square grid of neurons
    som = SOM(l, l)

    rm = RegionalModel(som, model_alg)
    rm.fit(X=x_tr, Y=y_tr, verboses=0, SOM_params=som_params, cluster_params=cluster_params)

    return rm


def _train_lssvm(x_tr, y_tr, case, **kwargs):
    clf = LSSVM(kernel='rbf', **case)
    clf.fit(x_tr, y_tr)

    return clf


def _train_local_lssvm(x_tr, y_tr, case, **kwargs):
    cluster_params = kwargs["cluster_params"]
    model_alg      = LSSVM(kernel='rbf', **case)

    lm = LocalModel(ClusterAlg=KMeans, ModelAlg=model_alg)
    lm.fit(x_tr, y_tr, Cluster_params=cluster_params)

    return lm


def _train_regional_lssvm(x_tr, y_tr, case, **kwargs):
    cluster_params = kwargs["cluster_params"]
    som_params     = kwargs["som_params"]

    model_alg = LSSVM(kernel='rbf', **case)
    return _train_regional_model(x_tr, y_tr, cluster_params, model_alg, som_params)


def _train_regional_ols(x_tr, y_tr, case, **kwargs):
    cluster_params = kwargs["cluster_params"]
    som_params     = kwargs["som_params"]

    model_alg = LinearRegression()
    return _train_regional_model(x_tr, y_tr, cluster_params, model_alg, som_params)


def search_best_hp(X_tr_norm, y_train, hps_cases, train_func, **kwargs):
    """
        5-fold stratified cross-validation for hyperparameter optimization
    """
    n_folds   = 5
    n_cases   = len(hps_cases)
    cv_scores = [0] * n_cases

    y_temp = process_labels(y_train)

    for i in range(n_cases):
        skf = StratifiedKFold(n_splits=n_folds)
        acc = [0] * n_folds
        count = 0

        for tr_index, val_index in skf.split(X_tr_norm, y_temp):  # train/validation split
            x_tr, x_val = X_tr_norm[tr_index], X_tr_norm[val_index]
            y_tr, y_val = y_train[tr_index], y_train[val_index]

            # Train the model on the train set
            clf = train_func(x_tr, y_tr, hps_cases[i], **kwargs)

            # Eval model accuracy on validation set
            acc[count] = accuracy_score(y_val, clf.predict(x_val))
            count += 1
        # apply objective function to cv accuracies
        cv_scores[i] = f_o(acc)

    # the best hyperparameters are the ones that maximize the objective function
    best_hps = hps_cases[np.argmax(cv_scores)]

    return best_hps, cv_scores


class HyperOptimization:

    @classmethod
    def _default_file_name(cls, dataset_name, random_state, model_name):
        return f"{PRE_PROCESS_PATH}/optimal-hypers/{model_name} - {dataset_name} - Random state {random_state}.pickle"

    @classmethod
    def load_optimal_hyperparams(cls, dataset_name, random_state, model_name):
        file_path   = cls._default_file_name(dataset_name, random_state, model_name)
        loaded_data = FileUtils.load_pickle_file(file_path)

        return loaded_data

    @classmethod
    def save_optimal_hyperparams(cls, opt_params, dataset_name, random_state, model_name):
        file_path = cls._default_file_name(dataset_name, random_state, model_name)
        FileUtils.save_pickle_file(data=opt_params, file_path=file_path)

        # FileUtils.save_pickle_file(
        #     data={"best_hps": best_hps, "best_hps_list": best_hps_list, "best_k": best_k, "suggestions": suggestions,
        #           "validation_scores": validation_scores, "som_tr": som_tr}, file_path=file_path)

    @classmethod
    def search_best_params(cls, model_name, search_func, dataset_name, random_state, X_tr_norm, y_train, hps_cases):
        loaded_opt_params = cls.load_optimal_hyperparams(dataset_name, random_state, model_name)

        if loaded_opt_params:
            return loaded_opt_params
        else:
            opt_params = search_func(X_tr_norm, y_train, hps_cases)
            cls.save_optimal_hyperparams(opt_params, dataset_name, random_state, model_name)
            return opt_params

    @classmethod
    def search_lssvm_best_hp(cls, X_tr_norm, y_train, hps_cases):
        best_hps, cv_scores = search_best_hp(X_tr_norm, y_train, hps_cases, train_func=_train_lssvm, **{})

        return best_hps

    @classmethod
    def search_local_lssvm_best_hp(cls, X_tr_norm, y_train, hps_cases):
        y_temp = process_labels(y_train)

        N = len(X_tr_norm)
        ks = np.arange(2, int(N ** (1 / 2)) + 1).tolist()  # 2 to sqrt(N)

        suggestions = get_k_opt_suggestions(X_tr_norm, y_temp, ks, cluster_val_metrics)
        unique_suggestions = np.unique(list(suggestions.values())).tolist()
        n_uniques = len(unique_suggestions)

        validation_scores = np.empty((n_uniques, 2))  # [k, cv_score]
        best_hps_list = [{}] * n_uniques
        count_v = 0

        for k in unique_suggestions:  # para cada proposta de k_{opt}
            cluster_params = {'n_clusters': k, 'n_init': 10, 'init': 'random'}
            best_hps, cv_scores = search_best_hp(X_tr_norm, y_train, hps_cases, _train_local_lssvm,
                                                 cluster_params=cluster_params)

            # The best hyperparameters are the ones that maximize the objective function
            best_hps_list[count_v] = hps_cases[np.argmax(cv_scores)]
            validation_scores[count_v, :] = [k, np.amax(cv_scores)]
            count_v += 1

        # k_opt as the one with best cv_score and smallest value
        best_k = int(validation_scores[validation_scores.argmax(axis=0)[1], 0])

        # the best hyperparameters are the ones that maximize the objective function
        best_hps = best_hps_list[validation_scores.argmax(axis=0)[1]]

        return best_hps, best_hps_list, best_k, suggestions, validation_scores

    @classmethod
    def search_regional_best_hp(cls, X_tr_norm, y_train, hps_cases, train_func):
        y_temp = process_labels(y_train)

        # Fit the SOM in the whole train set so we can get suggestion for number of clusters in K-Means
        N = len(X_tr_norm)
        l = int((5 * N ** .5) ** .5)  # size of square grid of neurons
        som_tr = SOM(l, l)
        som_tr.fit(X_tr_norm, **default_som_params)

        C = l ** 2  # number of SOM neurons in the 2D grid
        ks = np.arange(2, int(C ** (1 / 2)) + 1).tolist()  # 2 to sqrt(C)

        suggestions = get_k_opt_suggestions(som_tr.neurons, np.empty(C), ks, regional_cluster_val_metrics)
        unique_suggestions = np.unique(list(suggestions.values())).tolist()
        n_suggestions = len(unique_suggestions)

        validation_scores = np.empty((n_suggestions, 2))  # [k, cv_score]
        best_hps_list = [{}] * n_suggestions
        count_v = 0

        for k in unique_suggestions:  # para cada proposta de k_{opt}
            cluster_params = {'n_clusters': k, 'n_init': 10, 'init': 'random'}

            best_hps, cv_scores = search_best_hp(X_tr_norm, y_train, hps_cases, train_func,
                                                 cluster_params=cluster_params, som_params=default_som_params)

            # The best hyperparameters are the ones that maximize the objective function
            best_hps_list[count_v]        = hps_cases[np.argmax(cv_scores)]
            validation_scores[count_v, :] = [k, np.amax(cv_scores)]
            count_v += 1

        # k_opt as the one with best cv_score and smallest value
        best_k = int(validation_scores[validation_scores.argmax(axis=0)[1], 0])

        # the best hyperparameters are the ones that maximize the objective function
        best_hps = best_hps_list[validation_scores.argmax(axis=0)[1]]

        return best_hps, best_hps_list, best_k, suggestions, validation_scores, som_tr

    @classmethod
    def search_regional_ols_best_hp(cls, X_tr_norm, y_train, hps_cases):
        hps_cases = [None]
        best_hps, best_hps_list, best_k, suggestions, validation_scores, som_tr = \
            cls.search_regional_best_hp(X_tr_norm, y_train, hps_cases, train_func=_train_regional_ols)

        return best_k, som_tr

    @classmethod
    def search_regional_lssvm_best_hp(cls, X_tr_norm, y_train, hps_cases):
        return cls.search_regional_best_hp(X_tr_norm, y_train, hps_cases, train_func=_train_regional_lssvm)