import numpy as np
import plotly.graph_objects as go

from load_dataset import datasets

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics

from devcode.utils.metrics import dunn_fast, FPE, AIC, BIC, MDL, FPE
from devcode.utils import scale_feat, dummie2multilabel

from IPython.core.display import display, HTML


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


def evaluate_clusters(test_size, scaleType):
    for dataset_name in datasets:
        X = datasets[dataset_name]['features'].values
        Y = datasets[dataset_name]['labels'].values

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=np.unique(Y, axis=1),
                                                            test_size=test_size, random_state=0)
        # scaling features
        X_tr_norm, X_ts_norm = scale_feat(X_train, X_test, scaleType=scaleType)

        # solving multilabel problem in wall-following data set
        y_temp = y_train

        if y_train.ndim == 2:
            if y_train.shape[1] >= 2: y_temp = dummie2multilabel(y_train)

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


def find_optimal_k_per_dataset(test_size, scaleType):
    for ds_name in datasets:
        print(ds_name)

        X = datasets[ds_name]['features'].values
        Y = datasets[ds_name]['labels'].values

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=np.unique(Y, axis=1),
                                                            test_size=test_size, random_state=0)
        # scaling features
        X_tr_norm, X_ts_norm = scale_feat(X_train, X_test, scaleType=scaleType)

        # solving multilabel problem in wall-following data set
        y_temp = y_train
        if y_train.ndim == 2:
            if y_train.shape[1] >= 2: y_temp = dummie2multilabel(y_train)

        N = len(X_tr_norm)
        ks = np.arange(2, int(N ** (1 / 2)) + 1).tolist()
        print(get_k_opt_suggestions(X_tr_norm, y_temp, ks, cluster_val_metrics))
        print('\n')


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


def cluster_metrics_analysis(df, result_key):
    """

    Parameters
    ----------
    df           [dict]: dict containing multiple results
    result_key [String]: key from df dict

    Returns
    -------

    """
    for dataset_name in datasets:
        display(HTML('<center><h1>' + dataset_name + '</h1></center>'))

        _k_optimal_histogram(df, dataset_name, result_key, show_flag=False, save_flag=False)
        _hitrate_histogram_per_metric(df, dataset_name, result_key, show_flag=True, save_flag=True)

        display(HTML('<hr>'))


def k_optimal_histogram_local_vs_regional(df):
    for dataset_name in datasets:
        display(HTML('<center><h1>' + dataset_name + '</h1></center>'))

        # get dataframe of the specific dataset
        df_local    = df['local'].loc[df['local']['dataset_name'] == dataset_name]
        df_regional = df['regional'].loc[df['regional']['dataset_name'] == dataset_name]

        animals = ['giraffes', 'orangutans', 'monkeys']
        fig = go.Figure(data=[
            go.Bar(
                name='L-LSSVM',
                x=df_local["$k_{opt}$ [CV]"].value_counts().index.tolist(),
                y=df_local["$k_{opt}$ [CV]"].value_counts().values
            ),
            go.Bar(
                name='R-LSSVM',
                x=df_regional["$k_{opt}$ [CV]"].value_counts().index.tolist(),
                y=df_regional["$k_{opt}$ [CV]"].value_counts().values
            )
        ])
        # Change the bar mode
        fig.update_layout(barmode='group')
        fig.update_layout(
            #         title = "Distribuição do k_opt para as {} rodadas no conjunto <b>{}</b> e modelagem <b>{}</b>".format(
            #             len(df_dataset), dataset_name, model_type),
            xaxis_title='Número de agrupamentos',
            yaxis_title='Frequência',
            bargap=0.4,  # gap between bars of adjacent location coordinates
        )
        fig.update_layout(legend=dict(x=.86, y=1))

        #         # boxplot to k_{opt}
        #         fig = go.Figure(data=[go.Histogram(
        #             x=df_dataset["$k_{opt}$ [CV]"].values.tolist(),
        #             xbins_size=1,
        #             marker_color='rgb(55, 83, 109)'
        #         )])

        fig.update_layout(
            margin=dict(l=20, r=5, t=5, b=20),
            #         paper_bgcolor="LightSteelBlue"
        )

        fig.show()
        fig.write_image("images/r_l-lssvm_k_opt_dist_{}.pdf".format(dataset_name))

        display(HTML('<hr>'))