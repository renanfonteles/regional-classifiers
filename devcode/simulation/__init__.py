import numpy as np
import pandas as pd

from IPython.core.display import display
from ipywidgets import HTML

from devcode import cm2acc
from devcode.analysis.clustering import _k_optimal_histogram, _hitrate_histogram_per_metric, k_optimal_hitrate_heatmap, \
    k_opt_hitrates_per_dataset, get_cluster_acronyms
from devcode.analysis.results import extract_case, define_result_row
from devcode.utils.evaluation import per_round_metrics
from devcode.utils.visualization import create_multiple_boxplots, set_figure, add_line, create_multiple_barcharts, \
    set_custom_bar_layout


class ResultProcessor:

    @classmethod
    def process_results_in_multiple_datasets(cls, datasets, df_results, process_func):
        """
            Process results from Pandas Data Frame.

            Parameters
            ----------
            datasets        [list]              : Dataset names
            df_results      [Pandas.DataFrame]  : Results saved in a experiment
            process_func    [Callable]          : Function to process the results in DataFrame

            Returns
            -------
        """
        for dataset_name in datasets:
            display(HTML('<center><h1>' + dataset_name + '</h1></center>'))
            process_func(df_results, dataset_name)
            display(HTML('<hr>'))

    @classmethod
    def extract_confusion_matrix_per_run(cls, df, col):
        confusion_matrices = df[col].values
        n_runs             = confusion_matrices.shape[0]

        rows = [np.fromstring(confusion_matrices[i][1:-1], sep=" ") for i in range(n_runs)]

        if any([row.size == 0 for row in rows]):
            rows = [np.frombuffer(eval(confusion_matrices[i]), dtype='int64') for i in range(n_runs)]

        return np.array(rows)

    @classmethod
    def extract_metric_per_set(cls, dataframe_dicts, ds_name, model_labels):
        """
            Dataset-wise extraction of train and test set results (i.e., accuracy)

            Parameters
            ----------
            dataframe_dicts
            ds_name
            model_labels

            Returns
            -------
        """
        acc_per_set = {}

        for classifier in model_labels:
            df_classifier = dataframe_dicts[classifier]
            df_dataset    = df_classifier.loc[df_classifier['dataset_name'] == ds_name]

            tr_cms = cls.extract_confusion_matrix_per_run(df_dataset, col="cm_tr")
            ts_cms = cls.extract_confusion_matrix_per_run(df_dataset, col="cm_ts")

            tr_accuracies, specifities, sensibilities, f1_scores = per_round_metrics(tr_cms, as_pct=True)
            ts_accuracies, specifities, sensibilities, f1_scores = per_round_metrics(ts_cms, as_pct=True)

            acc_per_set[classifier] = {'treino': tr_accuracies, 'teste': ts_accuracies}

        return acc_per_set

    @classmethod
    def compare_boxplot_per_set(cls, dataframes_dict, ds_name):
        set_dict     = {'treino': 'cm_tr', 'teste': 'cm_ts'}
        model_labels = dataframes_dict.keys()

        data = cls.extract_metric_per_set(dataframes_dict, ds_name, model_labels)

        accuracies_per_set = [[data[classifier][set_] for classifier in model_labels] for set_ in set_dict]

        x_labels_tr = [f"Train {key.upper()}-LSSVM" for key in dataframes_dict.keys()]
        x_labels_ts = [f"Test {key.upper()}-LSSVM" for key in dataframes_dict.keys()]

        tr_boxplots_per_set = create_multiple_boxplots(datas=accuracies_per_set[0], x_labels=x_labels_tr)
        ts_boxplots_per_set = create_multiple_boxplots(datas=accuracies_per_set[1], x_labels=x_labels_ts)

        fig = set_figure(data=(tr_boxplots_per_set + ts_boxplots_per_set), yaxis={"title": "Accuracy (%)"},
                         showlegend=False)

        y = [np.mean(data['global']['treino']), np.mean(data['local']['treino']),
             np.mean(data['regional']['treino']), np.mean(data['global']['treino'])]

        fig = add_line(fig, x_labels_tr + [x_labels_tr[0]], y)

        y = [np.mean(data['global']['teste']), np.mean(data['local']['teste']),
             np.mean(data['regional']['teste']), np.mean(data['global']['teste'])]

        fig = add_line(fig, x_labels_ts + [x_labels_ts[0]], y)

        fig.show()
        fig.write_image("images/ar-lssvm_{}.pdf".format(ds_name))

    @classmethod
    def compare_local_regional_k_optimal(cls, dataframes_dict, ds_name):
        df_local    = dataframes_dict['local'].loc[dataframes_dict['local']['dataset_name'] == ds_name]
        df_regional = dataframes_dict['regional'].loc[dataframes_dict['regional']['dataset_name'] == ds_name]

        x_datas = [df_local["$k_{opt}$ [CV]"].value_counts().index.tolist(),
                   df_regional["$k_{opt}$ [CV]"].value_counts().index.tolist()]

        y_datas = [df_local["$k_{opt}$ [CV]"].value_counts().values,
                   df_regional["$k_{opt}$ [CV]"].value_counts().values]

        barcharts = create_multiple_barcharts(x_datas, y_datas, names=["L-LSSVM", "R-LSSVM"])

        fig = set_figure(data=barcharts)
        fig = set_custom_bar_layout(fig)

        fig.show()
        fig.write_image("images/r_l-lssvm_k_opt_dist_{ds_name}.pdf")

    @classmethod
    def local_cluster_analysis(cls, dataframes_dict, ds_name):
        _k_optimal_histogram(dataframes_dict, ds_name, result_key="local", show_flag=False, save_flag=False)
        _hitrate_histogram_per_metric(dataframes_dict, ds_name, result_key="local", show_flag=True, save_flag=True)

    @classmethod
    def regional_cluster_analysis(cls, dataframes_dict, ds_name):
        _k_optimal_histogram(dataframes_dict, ds_name, result_key="regional", show_flag=False, save_flag=False)
        _hitrate_histogram_per_metric(dataframes_dict, ds_name, result_key="regional", show_flag=True, save_flag=True)

    @classmethod
    def overall_local_heatmap_cluster_analysis(cls, dataframes_dict, ds_names):
        df_local    = dataframes_dict["local"]

        hitrates_per_dataset, metric_names = k_opt_hitrates_per_dataset(df_local, ds_names)
        metric_acronyms = get_cluster_acronyms(metric_names)

        k_optimal_hitrate_heatmap(hitrate_data=hitrates_per_dataset, metric_names=metric_acronyms, ds_names=ds_names,
                                  cmap="coolwarm", file_path='images/fig1_local_heatmap.pdf')

    @classmethod
    def overall_regional_heatmap_cluster_analysis(cls, dataframes_dict, ds_names):
        df_regional = dataframes_dict["regional"]

        hitrates_per_dataset, metric_names = k_opt_hitrates_per_dataset(df_regional, ds_names)
        metric_acronyms = get_cluster_acronyms(metric_names)

        k_optimal_hitrate_heatmap(hitrate_data=hitrates_per_dataset, metric_names=metric_acronyms, ds_names=ds_names,
                                  cmap="viridis", file_path='images/fig2_regional_heatmap.pdf')

    @classmethod
    def extract_table_evalution_metrics(cls, datasets, df_results, header, exp_params=None, params_keys=None,
                                        only_accuracy=True):
        n_cols   = len(header)

        df_ds = {}

        exp_params = exp_params if exp_params else [None]
        n_params   = len(exp_params)

        for dataset_name in datasets:
            df = df_results.loc[df_results['dataset_name'] == dataset_name]  # get simulation results

            if df.size == 0:
                print(f"Experimentos não encontratos para base de dados {dataset_name}")
                continue

            count   = 0
            df_data = np.empty((n_params, n_cols))  # matriz que guardará resultados numéricos

            for params in exp_params:
                df_case = extract_case(df, params, params_keys=params_keys) if (params and params_keys) else df

                test_confusion_matrix = cls.extract_confusion_matrix_per_run(df_case, col="cm_ts")

                accuracies, specifities, sensibilities, f1_scores = per_round_metrics(test_confusion_matrix,
                                                                                      as_pct=True)

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
            # display(df_ds[dataset_name])
            # print('-' * 100, '\n' * 2)

        return df_ds

    @classmethod
    def merge_table_results(cls, dataframes_list, ds_names, header):
        result_per_df = [cls.extract_table_evalution_metrics(ds_names, df, header=header, only_accuracy=False)
                         for df in dataframes_list]

        results_per_dataset = [[df_dict[ds_name] for df_dict in result_per_df] for ds_name in ds_names]

        df_results_per_dataset = [pd.concat(ds_result, axis=0) for ds_result in results_per_dataset]

        for df_dataset_result, ds_name in zip(df_results_per_dataset, ds_names):
            print(ds_name)
            df_dataset_result.set_axis(['Global', 'Local', 'Regional'], inplace=True)
            df_dataset_result = df_dataset_result.round(2)
            display(df_dataset_result)
        # print("t")

# from devcode.analysis.eigenvalues import eigenvalues_analysis_regional
#
# df_results = pd.read_csv("results/regional-results/ROLS - all - n_res=100 - 2019-07-10.csv")
# eigenvalues_analysis_regional(datasets, df["regional"])