import numpy as np
import pandas as pd

from code.utils import scale_feat, dummie2multilabel
from load_dataset import datasets

from code.models.lssvm import LSSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


def f_o(u):
    """ Objective function in validation strategy """
    return np.mean(u) - 2*np.std(u)


def eval_GLSSVM(filename, header, case, scaleType, test_size, hps_cases):
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