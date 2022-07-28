import datetime
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_feat(X_train, X_test, scaleType='min-max'):
    """ Function to scale data """

    if scaleType == 'min-max' or scaleType == 'std':
        X_tr_norm = np.copy(X_train) # making a copy to leave original available
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