import numpy as np


def eigenvalue_analysis(df_results):
    idx = 0
    eigvals_full = np.frombuffer(eval(df_results['eigenvalues'][idx]), dtype=df_results['eigenvalues_dtype'][idx])
    # print(eigvals_full)

    nan_indices  = np.argwhere(np.isnan(eigvals_full))
    eigvals_list = [None] * len(nan_indices)
    last_nan     = -1

    for i in range(len(nan_indices)):
        #     print(nan_indices[i][0])
        eigvals_list[i] = eigvals_full[last_nan + 1:nan_indices[i][0]]
        last_nan        = nan_indices[i][0]

    for eig in eigvals_list:
        print(eig)


