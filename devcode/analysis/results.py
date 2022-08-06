import numpy as np


def process_results(df_results):
    # from random import shuffle
    # shuffle(cases) # better estimation of remaining time

    from joblib import Parallel, delayed
    # data = Parallel(n_jobs=4, verbose=51)(delayed(method)(case) for case in reversed(cases))

    show_eigenvalues(df_results)
    show_metrics(df_results)


def show_eigenvalues(df_results):
    for i in range(len(df_results)):
        print(df_results['dataset_name'][i])
        print(
            np.frombuffer(eval(df_results['eigenvalues'][i]), dtype=df_results['eigenvalues_dtype'][i]).shape,
            df_results['eigenvalues_dtype'][i]
        )
        print(" ")


def show_metrics(df_results):
    for i in range(len(df_results)):
        print(df_results['dataset_name'][i])

        temp_tr = np.frombuffer(eval(df_results['cm_tr'][i]), dtype='int64')
        temp_ts = np.frombuffer(eval(df_results['cm_ts'][i]), dtype='int64')

        print("cm_tr:")
        print(temp_tr.reshape(int(len(temp_tr) ** (1 / 2)), -1))

        print("cm_ts:")
        print(temp_ts.reshape(int(len(temp_ts) ** (1 / 2)), -1))

        print("\n")
