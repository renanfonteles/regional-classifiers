import numpy as np
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from devcode.analysis.clustering import get_k_opt_suggestions, cluster_val_metrics, find_optimal_clusters

from devcode.models.local_learning import LocalModel
from devcode.models.lssvm import LSSVM
from devcode.utils import process_labels
from devcode.utils.evaluation import f_o


def search_lssvm_best_hp(X_tr_norm, y_train, hps_cases):
    """
    5-fold stratified cross-validation for hyperparameter optimization

    Parameters
    ----------
    X_tr_norm
    y_train
    hps_cases

    Returns
    -------

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
            # train the model on the train set
            clf = LSSVM(kernel='rbf', **hps_cases[i])
            clf.fit(x_tr, y_tr)
            # eval model accuracy on validation set
            acc[count] = accuracy_score(y_val, clf.predict(x_val))
            count += 1
        # apply objective function to cv accuracies
        cv_scores[i] = f_o(acc)

    # the best hyperparameters are the ones that maximize the objective function
    best_hps = hps_cases[np.argmax(cv_scores)]

    return best_hps


def search_local_lssvm_best_hp(X_tr_norm, y_train, hps_cases):
    y_temp = process_labels(y_train)

    N  = len(X_tr_norm)
    ks = np.arange(2, int(N ** (1 / 2)) + 1).tolist()  # 2 to sqrt(N)
    suggestions = get_k_opt_suggestions(X_tr_norm, y_temp, ks, cluster_val_metrics)

    unique_suggestions = np.unique(list(suggestions.values())).tolist()
    n_uniques = len(unique_suggestions)

    validation_scores = np.empty((n_uniques, 2))  # [k, cv_score]
    best_hps_list     = [{}] * n_uniques
    count_v = 0

    for k in unique_suggestions:  # para cada proposta de k_{opt}
        # 5-fold stratified cross-validation for hyperparameter optimization
        n_cases = len(hps_cases)
        cv_scores = [0] * n_cases
        for i in range(n_cases):
            skf   = StratifiedKFold(n_splits=5)
            acc   = [0] * 5
            count = 0

            # Train/validation split
            for tr_index, val_index in skf.split(X_tr_norm, y_temp):
                x_tr, x_val = X_tr_norm[tr_index], X_tr_norm[val_index]
                y_tr, y_val = y_train[tr_index], y_train[val_index]

                # Train the model on the train set
                cluster_params = {'n_clusters': k, 'n_init': 10, 'init': 'random'}
                model_alg      = LSSVM(kernel='rbf', **hps_cases[i])

                lm = LocalModel(ClusterAlg=KMeans, ModelAlg=model_alg)
                lm.fit(x_tr, y_tr, Cluster_params=cluster_params)

                # eval model accuracy on validation set
                acc[count] = accuracy_score(y_val, lm.predict(x_val))
                count += 1

            # apply objective function to cv accuracies
            cv_scores[i] = f_o(acc)

        # the best hyperparameters are the ones that maximize the objective function
        best_hps_list[count_v]        = hps_cases[np.argmax(cv_scores)]
        validation_scores[count_v, :] = [k, np.amax(cv_scores)]
        count_v += 1

    # k_opt as the one with best cv_score and smallest value
    best_k = int(validation_scores[validation_scores.argmax(axis=0)[1], 0])

    # the best hyperparameters are the ones that maximize the objective function
    best_hps = best_hps_list[validation_scores.argmax(axis=0)[1]]

    return best_hps, best_hps_list, best_k, suggestions, validation_scores


