import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
from IPython.core.display import display, HTML

from devcode.utils.visualization import render_boxplot_train_test, render_boxplot_with_histogram_train_test


def plot_boxplots_comparison(datasets, df_model, model_name, enable_hist=False):
    for ds_name in datasets:
        display(HTML('<center><h1>' + ds_name + '</h1></center>'))

        df_dataset = df_model.loc[df_model['dataset_name'] == ds_name]

        results = {'acc_train': None, 'acc_test': None}

        train_series, test_series = df_dataset['cm_tr'].values, df_dataset['cm_ts'].values

        n_exp = len(train_series)
        acc_train = [None] * n_exp
        acc_test  = [None] * n_exp

        for i in range(n_exp):
            # Train accuracy
            temp_tr      = np.frombuffer(eval(train_series[i]), dtype='int64')
            cm_tr        = temp_tr.reshape(int(len(temp_tr) ** (1 / 2)), -1)
            acc_train[i] = np.trace(cm_tr) / np.sum(cm_tr)

            # Test accuracy
            temp_ts      = np.frombuffer(eval(test_series[i]), dtype='int64')
            cm_ts        = temp_ts.reshape(int(len(temp_ts) ** (1 / 2)), -1)
            acc_test[i]  = np.trace(cm_ts) / np.sum(cm_ts)

        results['acc_train'] = acc_train
        results['acc_test']  = acc_test

        plot_func = _single_boxplot_with_histogram_comparison if enable_hist else _single_boxplot_comparison
        plot_func(ds_name, model_name, acc_train, acc_test)

        display(HTML('<hr>'))


def _single_boxplot_comparison(ds_name, model_name, tr_data, ts_data):
    title    = f"{model_name} accuracy distribution in dataset [<b>{ds_name}</b>]"
    render_boxplot_train_test(tr_data, ts_data, title, metric_name="Accuracy")


def _single_boxplot_with_histogram_comparison(ds_name, model_name, tr_data, ts_data):
    bin_sizes = {
        'pk'    : .015,
        'vc2c'  : .015,
        'vc3c'  : .0145,
        'wf24f' : .0015,
        'wf2f'  : .00072,
        'wf4f'  : .0015
    }

    bin_size = bin_sizes.get(ds_name)
    title    = f"{model_name} accuracy distribution in dataset [<b>{ds_name}</b>]"
    render_boxplot_with_histogram_train_test(tr_data, ts_data, title, metric_name="Accuracy", bin_size=bin_size)


def plot_eigenvalues(eigenvalues, dataset_name, comb, freq, show_flag=False):
    x = eigenvalues.real.tolist()
    y = eigenvalues.imag.tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram2dContour(x=x, y=y, colorscale='Hot', reversescale=True, xaxis='x', yaxis='y')
    )
    fig.add_trace(go.Scatter(x=x, y=y, xaxis='x', yaxis='y', mode='markers',
                             marker=dict(color='rgba(0,0,0,0.4)', size=6)))

    fig.add_trace(go.Histogram(y=y, xaxis='x2', marker=dict(color='rgba(0,0,0,1)')))
    fig.add_trace(go.Histogram(x=x, yaxis='y2', marker=dict(color='rgba(0,0,0,1)')))

    title = "Eigenvalues in <b>{}</b> dataset with ".format(dataset_name)
    title += "<b>k_opt={}; gamma={:.2E}; sigma={:.2E}</b>".format(comb[0], comb[1], comb[2])
    title += " [<b>{} instances</b>]".format(freq)
    fig.update_layout(
        title=title,
        xaxis=dict(zeroline=False, showgrid=False,
                   domain=[0, 0.85],
                   title='Real part'
                   ),
        yaxis=dict(zeroline=False,
                   domain=[0, 0.85],
                   title='Imaginary part'
                   ),
        xaxis2=dict(zeroline=False, showgrid=False,
                    domain=[0.85, 1]
                    ),
        yaxis2=dict(zeroline=False, showgrid=False,
                    domain=[0.85, 1]
                    ),
        bargap=0,
        hovermode='closest',
        showlegend=False
    )

    if show_flag:
        fig.show()
