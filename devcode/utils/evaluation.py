import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from devcode import LocalModel
from devcode.analysis.clustering import cluster_val_metrics, get_k_opt_suggestions
from devcode.models.local_learning import BiasModel
from devcode.utils import scale_feat, dummie2multilabel


from devcode.models.lssvm import LSSVM
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from pathlib import Path


def f_o(u):
    """ Objective function in validation strategy """
    return np.mean(u) - 2*np.std(u)


def eval_GLSSVM(datasets, filename, header, case, scaleType, test_size, hps_cases):
    """ Global Least-Squares Support Vector Machine """
    my_file = Path(filename)

    if not my_file.is_file():  # compute if it doesn't exists
        dataset_name = case['dataset_name']
        random_state = case['random_state']

        X = datasets[dataset_name]['features'].values
        Y = datasets[dataset_name]['labels'].values

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=np.unique(Y, axis=1),
                                                            test_size=test_size,
                                                            random_state=random_state)
        # scaling features
        X_tr_norm, X_ts_norm = scale_feat(X_train, X_test, scaleType=scaleType)

        # solving multilabel problem in wall-following data set
        y_temp = y_train
        if y_train.ndim == 2:
            if y_train.shape[1] >= 2:
                y_temp = dummie2multilabel(y_train)

        # 5-fold stratified cross-validation for hyperparameter optimization
        n_folds = 5
        n_cases = len(hps_cases)
        cv_scores = [0] * n_cases
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

        # fit the model on best hyperparameters
        clf = LSSVM(kernel='rbf', **best_hps)
        clf.fit(X_tr_norm, y_train)

        # make predictions and evaluate model
        y_pred_tr, y_pred_ts = clf.predict(X_tr_norm), clf.predict(X_ts_norm)
        cm_tr = confusion_matrix(dummie2multilabel(y_train),
                                 dummie2multilabel(y_pred_tr))
        cm_ts = confusion_matrix(dummie2multilabel(y_test),
                                 dummie2multilabel(y_pred_ts))

        # getting eigenvalues of kernel matrix
        K = clf.kernel(X_tr_norm, X_tr_norm)
        eig_vals = np.linalg.eigvals(K)

        data = np.array([dataset_name, random_state,
                         best_hps['gamma'], best_hps['sigma'],
                         eig_vals.tostring(), eig_vals.dtype,
                         cm_tr.tostring(), cm_ts.tostring()]).reshape(1, -1)

        results_df = pd.DataFrame(data, columns=header)
        results_df.to_csv(filename, index=False)  # saving results in csv fil


def eval_LLSSVM(datasets, filename, header, case, scaleType, test_size, hps_cases):
    """ Local Least-Squares Support Vector Machine """
    my_file = Path(filename)

    if not my_file.is_file():  # compute if it doesn't exists
        dataset_name = case['dataset_name']
        random_state = case['random_state']

        X = datasets[dataset_name]['features'].values
        Y = datasets[dataset_name]['labels'].values

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=np.unique(Y, axis=1),
                                                            test_size=test_size, random_state=random_state)
        # scaling features
        X_tr_norm, X_ts_norm = scale_feat(X_train, X_test, scaleType=scaleType)

        # solving multilabel problem in wall-following data set
        y_temp = y_train
        if y_train.ndim == 2:
            if y_train.shape[1] >= 2: y_temp = dummie2multilabel(y_train)

        N = len(X_tr_norm)
        ks = np.arange(2, int(N ** (1 / 2)) + 1).tolist()  # 2 to sqrt(N)
        suggestions = get_k_opt_suggestions(X_tr_norm, y_temp, ks, cluster_val_metrics)
        unique_suggestions = np.unique(list(suggestions.values())).tolist()

        validation_scores = np.empty((len(unique_suggestions), 2))  # [k, cv_score]
        best_hps_list = [{}] * len(unique_suggestions)
        count_v = 0

        for k in unique_suggestions:  # para cada proposta de k_{opt}
            # 5-fold stratified cross-validation for hyperparameter optimization
            n_cases = len(hps_cases)
            cv_scores = [0] * n_cases
            for i in range(n_cases):
                skf = StratifiedKFold(n_splits=5)
                acc = [0] * 5
                count = 0

                # train/validation split
                for tr_index, val_index in skf.split(X_tr_norm, y_temp):
                    x_tr, x_val = X_tr_norm[tr_index], X_tr_norm[val_index]
                    y_tr, y_val = y_train[tr_index], y_train[val_index]

                    # train the model on the train set
                    cluster_params = {'n_clusters': k, 'n_init': 10, 'init': 'random'}
                    model_alg = LSSVM(kernel='rbf', **hps_cases[i])
                    lm = LocalModel(ClusterAlg=KMeans, ModelAlg=model_alg)

                    lm.fit(x_tr, y_tr, Cluster_params=cluster_params)

                    # eval model accuracy on validation set
                    acc[count] = accuracy_score(y_val, lm.predict(x_val))
                    count += 1

                # apply objective function to cv accuracies
                cv_scores[i] = f_o(acc)

            # the best hyperparameters are the ones that maximize the objective function
            best_hps_list[count_v] = hps_cases[np.argmax(cv_scores)]
            validation_scores[count_v, :] = [k, np.amax(cv_scores)]
            count_v += 1

        # k_opt as the one with best cv_score and smallest value
        best_k = int(validation_scores[validation_scores.argmax(axis=0)[1], 0])
        #         print("best_k")
        #         print(best_k)

        # the best hyperparameters are the ones that maximize the objective function
        best_hps = best_hps_list[validation_scores.argmax(axis=0)[1]]
        #         print("best_hps")
        #         print(best_hps)

        # fit the model on best global hyperparameters
        cluster_params = {'n_clusters': best_k, 'n_init': 10, 'init': 'random'}
        model_alg = LSSVM(kernel='rbf', **best_hps)
        lm = LocalModel(ClusterAlg=KMeans, ModelAlg=model_alg)
        lm.fit(X_tr_norm, y_train, Cluster_params=cluster_params)

        # make predictions and evaluate model
        y_pred_tr, y_pred_ts = lm.predict(X_tr_norm), lm.predict(X_ts_norm)
        cm_tr = confusion_matrix(dummie2multilabel(y_train),
                                 dummie2multilabel(y_pred_tr))
        cm_ts = confusion_matrix(dummie2multilabel(y_test),
                                 dummie2multilabel(y_pred_ts))

        # getting eigenvalues of kernel matrices
        n_lssvms = 0  # counter for LSSVM models
        models_idx = []
        for i in range(len(lm.models)):
            if isinstance(lm.models[i], LSSVM):
                n_lssvms += 1
                models_idx.append(i)

        eigvals_list = [None, np.array([np.nan])] * n_lssvms
        count = 0
        for i in models_idx:
            x_region = X_tr_norm[lm.ClusterAlg.labels_ == i]
            K = lm.models[i].kernel(x_region, x_region)
            temp = np.linalg.eigvals(K)
            eigvals_list[count] = temp  # .tostring()
            count += 2

        eigvals = np.concatenate(eigvals_list, axis=0)

        k_opt = len(lm.models)
        n_empty_regions = len(lm.empty_regions)
        n_homogeneous_regions = 0
        for i in range(len(lm.models)):
            if isinstance(lm.models[i], BiasModel):
                n_homogeneous_regions += 1

        #         print("suggestions")
        #         print(suggestions)

        # Organizing suggestion of the cluster metrics
        temp = [np.nan] * 2 * len(cluster_val_metrics)
        count = 0
        for metric in cluster_val_metrics:
            temp[count] = best_hps_list[
                np.where(validation_scores[:, 0] == suggestions[metric['name']])[0][0]]['gamma']
            temp[count + 1] = best_hps_list[
                np.where(validation_scores[:, 0] == suggestions[metric['name']])[0][0]]['sigma']
            count += 2

        data = np.array(
            [dataset_name, random_state,
             n_empty_regions, n_homogeneous_regions] + \
            [best_hps['gamma'], best_hps['sigma']] + \
            temp + \
            [k_opt] + \
            [value for value in list(suggestions.values())] + \
            [validation_scores[
                 np.where(validation_scores[:, 0] == suggestions[metric['name']])[0][0],
                 1] for metric in cluster_val_metrics] + \
            [eigvals.tostring(), eigvals.dtype, cm_tr.tostring(), cm_ts.tostring()]
        ).reshape(1, -1)

        #         print(' ')
        #         print("cm2acc(cm_tr)")
        #         print(cm2acc(cm_tr))
        #         print("cm2acc(cm_ts)")
        #         print(cm2acc(cm_ts))

        results_df = pd.DataFrame(data, columns=header)
        results_df.to_csv(filename, index=False)  # saving results in csv file

        # saving clusters to .csv
        clusters = lm.ClusterAlg.cluster_centers_
        clt_filename = "./temp_llssvm_cbic/clusters/L-LSSVM - {}.csv".format(case)
        pd.DataFrame(clusters).to_csv(clt_filename, header=None, index=None)


# results_df = eval_LLSSVM(cases[0])


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from functools import partial
from multiprocessing import Pool


# def evalRLM(datasets, case):
#     dataset_name = case['dataset_name']
#     random_state = case['random_state']
#     som_params = case['som_params']
#
#     X = datasets[dataset_name]['features'].values
#     Y = datasets[dataset_name]['labels'].values
#
#     # Train/Test split
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
#     # scaling features
#     X_tr_norm, X_ts_norm = scale_feat(X_train, X_test, scaleType=scaleType)
#
#     N = len(X_tr_norm)  # number of datapoints in the train split
#     l = ceil((5 * N ** .5) ** .5)  # side length of square grid of neurons
#
#     som = SOM(l, l)
#
#     C = l ** 2  # number of SOM neurons in the 2D grid
#     k_values = [i for i in range(2, ceil(np.sqrt(C)))]  # 2 to sqrt(C)
#     cluster_params = {
#         'n_clusters': {'metric': DB,  # when a dictionary is pass a search begins
#                        'criteria': np.argmin,  # search for smallest DB score
#                        'k_values': k_values},  # around the values provided in 'k_values'
#         'n_init': 10,  # number of initializations
#         'init': 'random'
#         # 'n_jobs':     0
#     }
#
#     linearModel = linear_model.LinearRegression()
#
#     rlm = RegionalModel(som, linearModel)
#     rlm.fit(X=X_tr_norm, Y=y_train,
#             SOM_params=som_params,
#             Cluster_params=cluster_params)
#
#     # Evaluating in the train set
#     y_tr_pred = rlm.predict(X_tr_norm)
#     y_tr_pred = np.round(np.clip(y_tr_pred, 0, 1))  # rounding prediction numbers
#
#     cm_tr = confusion_matrix(dummie2multilabel(y_train),
#                              dummie2multilabel(y_tr_pred)
#                              ).reshape(-1)  # matrix => array
#
#     # Evaluating in the test set
#     y_ts_pred = rlm.predict(X_ts_norm)
#     y_ts_pred = np.round(np.clip(y_ts_pred, 0, 1))  # rounding prediction numbers
#
#     cm_ts = confusion_matrix(dummie2multilabel(y_test),
#                              dummie2multilabel(y_ts_pred)
#                              ).reshape(-1)  # matrix => array
#
#     data = [dataset_name, random_state] + list(som_params.values()) + [cm_tr] + [cm_ts]
#     return data


def cm2acc(cm):
    return np.trace(cm) / np.sum(cm)


def cm2sen(cm):
    sensitivity = 0
    size = len(cm)

    if size == 2:  # binary classification
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        sensitivity = tp / (tp + fn)

    else:  # multiclass classification
        sens = [None] * size
        for i in range(size):
            tn = cm[i][i]
            fp = np.sum(cm[i, :]) - tn
            fn = np.sum(cm[:, i]) - tn
            tp = np.trace(cm) - tn
            sens[i] = tp / (tp + fn)

        sensitivity = np.mean(sens)

    return sensitivity


def cm2esp(cm):
    specificity = 0
    size = len(cm)

    if size == 2:  # binary classification
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        specificity = tn / (tn + fp)

    else:  # multiclass classification
        spec = [None] * size
        for i in range(size):
            tn = cm[i][i]
            fp = np.sum(cm[i, :]) - tn
            fn = np.sum(cm[:, i]) - tn
            tp = np.trace(cm) - tn
            spec[i] = tn / (tn + fp)

        specificity = np.mean(spec)

    return specificity


def cm2f1(cm):
    f1 = 0
    size = len(cm)

    if size == 2:  # binary classification
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

    else:  # multiclass classification
        temp_f1 = [None] * size
        for i in range(size):
            tn = cm[i][i]
            fp = np.sum(cm[i, :]) - tn
            fn = np.sum(cm[:, i]) - tn
            tp = np.trace(cm) - tn
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            temp_f1[i] = 2 * (precision * recall) / (precision + recall)

        f1 = np.mean(temp_f1)

    return f1