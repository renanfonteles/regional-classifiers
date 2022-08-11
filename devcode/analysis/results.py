import numpy as np
import pandas as pd
from IPython.core.display import display

from devcode.utils.evaluation import per_round_metrics


def process_results(df_results):
    if 'eigenvalues' in df_results:
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


def define_result_row(params, params_keys, metrics, stats_functions):
    n_metrics = len(metrics)
    params_values = [params[key] for key in params_keys] if (params and params_keys) else []
    metrics_stats = [stats_func(metrics[i]) for i in range(n_metrics) for stats_func in stats_functions[i]]

    result_row = np.matrix(params_values + metrics_stats)
    return result_row


def extract_case(df, params, params_keys):
    _filter_params = []

    for key in params_keys:
        key_df = f"$\{key}$" if (key == "gamma" or key == "sigma") else key
        # if df[key_df].dtype == 'float64':
        #     df[key_df]  = df[key_df].round(decimals=2)
        #     params[key] = round(params[key], 2)

        _filter_params.append((df[key_df] == params[key]))

    combined_filter = _filter_params[0]

    for _filter in _filter_params:
        combined_filter = combined_filter & _filter

    df_case = df.loc[combined_filter]

    return df_case


def results_per_dataset(datasets, df_results, header, exp_params=None, params_keys=None, only_accuracy=True):
    n_cols   = len(header)

    df_ds = {}

    exp_params = exp_params if exp_params else [None]
    n_params   = len(exp_params)

    for dataset_name in datasets:  # For this specific dataset
        print(dataset_name)
        df = df_results.loc[df_results['dataset_name'] == dataset_name]  # get simulation results

        if df.size == 0:
            print(f"Experimentos não encontratos para base de dados {dataset_name}")
            continue

        count   = 0
        df_data = np.empty((n_params, n_cols))  # matriz que guardará resultados numéricos

        for params in exp_params:
            df_case = extract_case(df, params, params_keys=params_keys) if (params and params_keys) else df

            cm      = df_case['cm_ts'].values
            n_runs  = cm.shape[0]

            rows = [np.fromstring(df_case['cm_ts'].values[i][1:-1], sep=" ") for i in range(n_runs)]

            if any([row.size == 0 for row in rows]):
                print("teste")
                rows = [np.frombuffer(eval(df['cm_ts'].values[i]), dtype='int64') for i in range(n_runs)]

            test_confusion_matrix = np.array(rows)

            accuracies, specifities, sensibilities, f1_scores = per_round_metrics(test_confusion_matrix)

            if only_accuracy:
                metrics         = [accuracies]
                stats_functions = [[np.mean, np.std]]
            else:
                metrics         = [accuracies, specifities, sensibilities, f1_scores]
                stats_functions = [[np.mean, np.std], [np.mean], [np.mean], [np.mean]]

            data_row = define_result_row(params, params_keys, metrics=metrics, stats_functions=stats_functions)

            df_data[count, :] = data_row

            count += 1

        df_ds[dataset_name] = pd.DataFrame(df_data, columns=header)
        display(df_ds[dataset_name])
        print('-' * 100, '\n' * 2)
