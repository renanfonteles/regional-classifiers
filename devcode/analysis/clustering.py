import numpy as np
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn import metrics

from devcode.utils.metrics import dunn_fast, FPE, AIC, BIC, MDL, FPE
from devcode.utils import process_labels, collect_data
from devcode.models.local_learning import BiasModel

from IPython.core.display import display, HTML


metrics_acronyms = {
    "Silhouette"                    : "SI",
    "Calinski-Harabasz"             : "CH",
    "Davies-Bouldin"                : "DB",
    "Dunn"                          : "DU",
    "Final Prediction Error"        : "FPE",
    "Akaike Information Criteria"   : "AIC",
    "Bayesian Information Criteria" : "BIC",
    "Minimum Description Length"    : "MDL",

    "Adjusted Rand Index"           : "ARI",
    "Adjusted Mutual Information"   : "AMI",
    "V-measure"                     : "VM",
    "Fowlkes-Mallows"               : "FM",
}


def get_cluster_acronyms(metric_names):
    return [metrics_acronyms[name] for name in metric_names]


# https://scikit-learn.org/stable/modules/clustering.html
cluster_val_metrics = [
    {
        'name'    : 'Adjusted Rand Index',
        'f'       : metrics.adjusted_rand_score,
        'get_best': np.argmax,
        'type'    : 'supervised'
    },
    {
        'name'    : 'Adjusted Mutual Information',
        'f'       : metrics.adjusted_mutual_info_score,
        'get_best': np.argmax,
        'type'    : 'supervised'
    },
    {
        'name'    : 'V-measure',
        'f'       : metrics.v_measure_score,
        'get_best': np.argmax,
        'type'    : 'supervised'
    },
    {
        'name'    : 'Fowlkes-Mallows',
        'f'       : metrics.fowlkes_mallows_score,
        'get_best': np.argmax,
        'type'    : 'supervised'
    },
    {
        'name'    : 'Silhouette',
        'f'       : metrics.silhouette_score,
        'get_best': np.argmax,
        'type'    : 'unsupervised'
    },
    {
        'name'    : 'Calinski-Harabasz',
        'f'       : metrics.calinski_harabasz_score,
        'get_best': np.argmax,
        'type'    : 'unsupervised'
    },
    {
        'name'    : 'Davies-Bouldin',
        'f'       : metrics.davies_bouldin_score,
        'get_best': np.argmin,
        'type'    : 'unsupervised'
    },
    {
        'name'    : 'Dunn',
        'f'       : dunn_fast,
        'get_best': np.argmax,
        'type'    : 'unsupervised'
    },
    {
        'name'    : 'Final Prediction Error',
        'f'       : FPE,
        'get_best': np.argmin,
        'type'    : 'unsupervised'
    },
    {
        'name'    : 'Akaike Information Criteria',
        'f'       : AIC,
        'get_best': np.argmin,
        'type'    : 'unsupervised'
    },
    {
        'name'    : 'Bayesian Information Criteria',
        'f'       : BIC,
        'get_best': np.argmin,
        'type'    : 'unsupervised'
    },
    {
        'name'    : 'Minimum Description Length',
        'f'       : MDL,
        'get_best': np.argmin,
        'type'    : 'unsupervised'
    },
]


def get_k_opt_suggestions(X, y, ks, cluster_metrics):
    """
        Return optimal k suggestion
    """
    results = {metric['name']: [None] * len(ks) for metric in cluster_metrics}
    for i in range(len(ks)):
        kmeans = KMeans(n_clusters=ks[i], init='random').fit(X)
        labels_true = y.ravel()
        labels_pred = kmeans.labels_

        temp = eval_cluster(cluster_metrics, X, labels_true, labels_pred)
        for name in temp:
            results[name][i] = temp[name]

    suggestions = {metric['name']: np.nan for metric in cluster_metrics}
    for i in range(len(cluster_metrics)):
        metric = cluster_metrics[i]
        suggestions[metric['name']] = ks[metric['get_best'](results[metric['name']])]

    return suggestions


def eval_cluster(cluster_metrics, X, labels_true, labels_pred):
    names = [metric['name'] for metric in cluster_metrics]
    results = {}

    for metric in cluster_metrics:
        if metric['type'] == 'supervised':
            results[metric['name']] = metric['f'](labels_true, labels_pred)
        else:
            results[metric['name']] = metric['f'](X, labels_pred)

    return results


def evaluate_clusters(datasets, test_size, scaleType):
    for dataset_name in datasets:
        X_tr_norm, y_train, X_ts_norm, y_test = collect_data(datasets, dataset_name, random_state=0,
                                                             test_size=test_size, scale_type=scaleType)

        y_temp = process_labels(y_train)

        N  = len(X_tr_norm)
        ks = np.arange(2, int(N ** (1 / 2)) + 1).tolist()

        n_ks = len(ks)

        results = {metric['name']: [None] * n_ks for metric in cluster_val_metrics}
        # print(results)

        from tqdm import tnrange, tqdm_notebook
        for i in tnrange(n_ks):
            kmeans = KMeans(n_clusters=ks[i], init='random', random_state=0).fit(X_tr_norm)
            labels_true = y_temp.ravel()
            labels_pred = kmeans.labels_

            temp = eval_cluster(cluster_val_metrics, X_tr_norm, labels_true, labels_pred)

            for name in temp:
                results[name][i] = temp[name]

        # Create traces
        fig = go.Figure()
        for name in results:
            fig.add_trace(go.Scatter(x=ks, y=results[name], mode='lines+markers', name=name))

        fig.update_layout(title_text="Clustering metrics x number of clusters [<b>{}</b>]".format(dataset_name),
                          xaxis_title='Number of clusters',
                          yaxis_title='Metrics'
                          )
        fig.show()

        suggestions_for_k = np.empty(len(cluster_val_metrics), dtype='int16')

        for i in range(len(cluster_val_metrics)):
            metric = cluster_val_metrics[i]
            suggestions_for_k[i] = ks[metric['get_best'](results[metric['name']])]

            print('Best k for {:30}: {}'.format(metric['name'], suggestions_for_k[i]))

        print(' ')
        print("Unique sugestions of k_opt = {}".format(np.unique(suggestions_for_k).tolist()))


def find_optimal_k_per_dataset(datasets, test_size, scaleType):
    for ds_name in datasets:
        X_tr_norm, y_train, X_ts_norm, y_test = collect_data(datasets, ds_name, random_state=0,
                                                             test_size=test_size, scale_type=scaleType)

        y_temp = process_labels(y_train)

        N = len(X_tr_norm)
        ks = np.arange(2, int(N ** (1 / 2)) + 1).tolist()
        print(get_k_opt_suggestions(X_tr_norm, y_temp, ks, cluster_val_metrics))
        print('\n')


def extract_hitrate_per_clustering_metric(df, ds_name):
    n_metrics = int((len(df.columns[6:-4]) - 1) / 4)

    metric_names = [' '] * n_metrics

    count = 0
    for text in list(df.columns[-4 - n_metrics:-4]):
        metric_names[count] = text[10:-1]
        count += 1

    # get dataframe of the specific dataset and k_opt
    temp = 6 + 2 * n_metrics
    df_dataset = df.loc[df['dataset_name'] == ds_name].iloc[:, temp:(temp + n_metrics + 1)]

    hitrates_per_metric = [0] * len(metric_names)
    for i in range(len(df_dataset)):
        for j in range(len(metric_names)):
            column_name = "$k_{opt}$" + " [{}]".format(metric_names[j])
            if df_dataset[column_name].values[i] == df_dataset["$k_{opt}$ [CV]"].values[i]:
                hitrates_per_metric[j] += 1

    return hitrates_per_metric, metric_names


def k_opt_hitrates_per_dataset(df, datasets):
    hitrates_values = [extract_hitrate_per_clustering_metric(df, ds_name) for ds_name in datasets]
    metric_names = hitrates_values[0][1]
    hitrates_per_dataset = [hit_value for hit_value, _ in hitrates_values]

    return hitrates_per_dataset, metric_names


def _hitrate_histogram_per_metric(df, dataset_name, result_key, show_flag=False, save_flag=False):
    for model_type in [result_key]:  # ['local', 'regional']:
        n_metrics = int((len(df[model_type].columns[6:-4]) - 1) / 4)

        metric_names = [' '] * n_metrics
        count = 0
        for text in list(df[model_type].columns[-4 - n_metrics:-4]):
            metric_names[count] = text[10:-1]
            count += 1

        # get dataframe of the specific dataset and k_opt
        temp = 6 + 2 * n_metrics
        df_dataset = df[model_type].loc[df[model_type]['dataset_name'] == dataset_name
                                        ].iloc[:, temp:(temp + n_metrics + 1)]

        x = metric_names
        y = [0] * len(metric_names)
        for i in range(len(df_dataset)):
            for j in range(len(metric_names)):
                column_name = "$k_{opt}$" + " [{}]".format(metric_names[j])
                if df_dataset[column_name].values[i] == df_dataset["$k_{opt}$ [CV]"].values[i]:
                    y[j] += 1

        fig = go.Figure()
        for i in range(len(x)):
            fig.add_trace(go.Bar(x=[x[i]], y=[y[i]], text=y[i], textposition='auto'))

        #     fig = go.Figure([go.Bar(
        #         x=x,
        #         y=y,
        #         text=y,
        #         textposition='auto',
        #         marker_color='lightsalmon'
        #     )]
        #     )
        fig.update_layout(
            #             title_text='Frequência de acerto da proposta ótima para cada métrica no '+ \
            #                         'conjunto <b>{}</b> e modelagem <b>{}</b>'.format(dataset_name, model_type),
            yaxis_title='Frequência de acerto do k_opt',
            showlegend=False
        )
        fig.update_layout(
            margin=dict(l=20, r=5, t=5, b=20),
            #             paper_bgcolor="LightSteelBlue",
        )

        if show_flag:
            fig.show()

        if save_flag:
            fig.write_image("images/r-lssvm_metrics-k_opt-hit-frequency_{}.pdf".format(dataset_name))


def _k_optimal_histogram(df, dataset_name, result_key, show_flag=False, save_flag=False):
    """

    Parameters
    ----------
    df
    dataset_name

    Returns
    -------

    """
    for model_type in [result_key]:  # ['local', 'regional']:
        n_metrics = int((len(df[model_type].columns[6:-4]) - 1) / 4)

        metric_names = [' '] * n_metrics
        count = 0
        for text in list(df[model_type].columns[-4 - n_metrics:-4]):
            metric_names[count] = text[10:-1]
            count += 1

        # Get dataframe of the specific dataset
        df_dataset = df[model_type].loc[df[model_type]['dataset_name'] == dataset_name]

        # Boxplot to k_{opt}
        fig = go.Figure(data=[go.Histogram(x=df_dataset["$k_{opt}$ [CV]"].values.tolist(), xbins_size=1,
                                           marker_color='rgb(55, 83, 109)')])
        fig.update_layout(
            xaxis_title='Número de agrupamentos [regiões locais]',
            yaxis_title='Frequência',
            bargap=0.1,  # gap between bars of adjacent location coordinates
        )
        #     fig.update_xaxes(range=[0, 4])
        fig.update_layout(
            margin=dict(l=20, r=5, t=5, b=20),
            #             paper_bgcolor="LightSteelBlue",
        )

        if show_flag:
            fig.show()

        if save_flag:
            fig.write_image("images/r-lssvm_k_opt_dist_{}.pdf".format(dataset_name))


def k_optimal_hitrate_heatmap(hitrate_data, metric_names, ds_names, cmap, file_path, show_flag=True, save_flag=True):
    from pandas import DataFrame
    import seaborn as sns
    import matplotlib.pyplot as plt

    data_np_reg = np.array(hitrate_data)

    df2 = DataFrame(data_np_reg, index=ds_names, columns=metric_names)

    ax2 = sns.heatmap(df2, annot=True, cmap=cmap, cbar_kws={'label': 'Hit-rate of $K_{opt}$'})
    ax2.set_ylabel('Datasets')
    ax2.set_xlabel('Indices acronyms')

    if show_flag:
        plt.show()

    if save_flag:
        plt.savefig(file_path)


def find_optimal_clusters(lm, suggestions, validation_scores, best_hps_list):
    k_opt           = len(lm.models)
    n_empty_regions = len(lm.empty_regions)
    n_homogeneous_regions = 0

    for i in range(len(lm.models)):
        if isinstance(lm.models[i], BiasModel):
            n_homogeneous_regions += 1

    # Organizing suggestion of the cluster metrics
    temp = [np.nan] * 2 * len(cluster_val_metrics)
    count = 0
    for metric in cluster_val_metrics:
        temp[count] = best_hps_list[
            np.where(validation_scores[:, 0] == suggestions[metric['name']])[0][0]]['gamma']
        temp[count + 1] = best_hps_list[
            np.where(validation_scores[:, 0] == suggestions[metric['name']])[0][0]]['sigma']
        count += 2

    empty_and_homo = [n_empty_regions, n_homogeneous_regions]
    valid_metrics  = [validation_scores[np.where(validation_scores[:, 0] == suggestions[metric['name']])[0][0], 1]
                      for metric in cluster_val_metrics]

    return empty_and_homo, valid_metrics


def organize_cluster_metrics(best_hps_list, validation_scores, suggestions):
    # Organizing suggestion of the cluster metrics
    temp = [np.nan] * 2 * len(cluster_val_metrics)
    count = 0
    for metric in cluster_val_metrics:
        temp[count] = best_hps_list[
            np.where(validation_scores[:, 0] == suggestions[metric['name']])[0][0]]['gamma']
        temp[count + 1] = best_hps_list[
            np.where(validation_scores[:, 0] == suggestions[metric['name']])[0][0]]['sigma']
        count += 2

    return temp