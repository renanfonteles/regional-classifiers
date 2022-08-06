import numpy as np
import plotly.graph_objects as go

from load_dataset import datasets

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics

from devcode.utils.metrics import dunn_fast, FPE, AIC, BIC, MDL, FPE
from devcode.utils import scale_feat, dummie2multilabel


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
