import itertools

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


def eigenvalues_analysis(datasets, df_model):
    for dataset_name in datasets:
        display(HTML('<center><h1>' + dataset_name + '</h1></center>'))
        print(" ")
        # get dataframe of the specific dataset
        df_dataset = df_model.loc[df_model['dataset_name'] == dataset_name]

        hps_names = ["$\gamma$", "$\sigma$"]
        hps_comb  = np.unique(df_dataset[hps_names].values, axis=0)  # hyperparam combinations
        #     print(hps_comb)

        for comb in hps_comb:
            df_comb = df_dataset.loc[(df_dataset[hps_names[0]] == comb[0]) & (df_dataset[hps_names[1]] == comb[1])]

            #         display(df_comb)

            eigen_string = df_comb['eigenvalues'].values
            dtypes = df_comb['eigenvalues_dtype'].values
            eigenvalues_list = [None] * len(dtypes)
            cond_list = np.empty(len(dtypes))  # conditioning
            for i in range(len(dtypes)):
                eigenvalues_list[i] = np.frombuffer(eval(eigen_string[i]), dtype=dtypes[i])
                modules = np.absolute(eigenvalues_list[i])
                cond_list[i] = np.amax(modules) / np.amin(modules)

            freq = len(eigenvalues_list)

            eigenvalues = np.concatenate(eigenvalues_list)
            cond_worst = np.amax(cond_list)

            #         x = eigenvalues.real.tolist()
            #         y = eigenvalues.imag.tolist()

            #         fig = go.Figure()
            #         fig.add_trace(
            #             go.Histogram2dContour(x=x, y=y, colorscale='Hot', reversescale=True, xaxis='x', yaxis='y')
            #         )
            #         fig.add_trace(
            #             go.Scatter(x=x, y=y, xaxis='x', yaxis='y', mode='markers',
            #                 marker=dict(color='rgba(0,0,0,0.4)', size=6)
            #             ))
            #         fig.add_trace(go.Histogram(y=y,
            #                 xaxis = 'x2',
            #                 marker = dict(
            #                     color = 'rgba(0,0,0,1)'
            #                 )
            #             ))
            #         fig.add_trace(go.Histogram(x=x,
            #                 yaxis = 'y2',
            #                 marker = dict(
            #                     color = 'rgba(0,0,0,1)'
            #                 )
            #             ))

            #         title = "Eigenvalues in <b>{}</b> dataset with ".format(dataset_name)
            #         title+="<b>gamma={:.2E}</b>; <b>sigma={:.2E}</b>".format(comb[0], comb[1])
            #         title+= " [<b>{} instances</b>]".format(freq)
            #         fig.update_layout(
            #             title = title,
            #             xaxis = dict(zeroline=False, showgrid=False,
            #                 domain = [0,0.85],
            #                 title='Real part'
            #             ),
            #             yaxis = dict(zeroline = False,
            #                 domain = [0,0.85],
            #                 title='Imaginary part'
            #             ),
            #             xaxis2 = dict(zeroline = False, showgrid = False,
            #                 domain = [0.85,1]
            #             ),
            #             yaxis2 = dict(zeroline = False, showgrid = False,
            #                 domain = [0.85,1]
            #             ),
            #             bargap = 0,
            #             hovermode = 'closest',
            #             showlegend = False
            #         )

            #         fig.show()

            print("case: ", end='')
            print("gamma = {:.2E}; sigma = {:.2E}".format(comb[0], comb[1]), end=' ')
            print("[{:}]".format(str(freq) + " instances"))
            print("dtypes = {}".format(np.unique(dtypes)))
            print("Worst conditioning: {:.2E}".format(cond_worst))
            print("-" * 55)

            print("\n")

        display(HTML('<hr>'))


def eigenvalues_analysis_local(datasets, df_model):
    for dataset_name in datasets:
        display(HTML('<center><h1>' + dataset_name + '</h1></center>'))
        print(" ")
        # get dataframe of the specific dataset
        df_dataset = df_model.loc[df_model['dataset_name'] == dataset_name]

        hps_names = ["$k_{opt}$ [CV]", "$\gamma_{opt}$ [CV]", "$\sigma_{opt}$ [CV]"]
        hps_comb = np.unique(df_dataset[hps_names].values, axis=0)  # hyperparam combinations
        for comb in hps_comb:
            df_comb = df_dataset.loc[(df_dataset[hps_names[0]] == comb[0]) &
                                     (df_dataset[hps_names[1]] == comb[1]) &
                                     (df_dataset[hps_names[2]] == comb[2])
                                     ]

            eigen_string = df_comb['eigenvalues'].values
            dtypes = df_comb['eigenvalues_dtype'].values
            eigenvalues_list = [None] * len(dtypes)
            cond_list = []  # np.empty(len(dtypes)) # conditioning
            for i in range(len(dtypes)):
                eigvals_full = np.frombuffer(eval(eigen_string[i]),
                                             dtype=dtypes[i])

                nan_indices = np.argwhere(np.isnan(eigvals_full))
                eigvals_list = [None] * len(nan_indices)
                last_nan = -1
                for j in range(len(nan_indices)):
                    #     print(nan_indices[i][0])
                    eigvals_list[j] = eigvals_full[last_nan + 1:nan_indices[j][0]]
                    last_nan = nan_indices[j][0]

                    modules = np.absolute(eigvals_list[j])
                    cond_list.append(np.amax(modules) / np.amin(modules))

                eigenvalues_list[i] = eigvals_list

            freq = len(eigenvalues_list)

            # list of lists to single list
            merged = list(itertools.chain(*eigenvalues_list))

            eigenvalues = np.concatenate(merged)
            cond_worst = np.amax(cond_list)

            #         x = eigenvalues.real.tolist()
            #         y = eigenvalues.imag.tolist()

            #         fig = go.Figure()
            #         fig.add_trace(
            #             go.Histogram2dContour(x=x, y=y, colorscale='Hot', reversescale=True, xaxis='x', yaxis='y')
            #         )
            #         fig.add_trace(
            #             go.Scatter(x=x, y=y, xaxis='x', yaxis='y', mode='markers',
            #                 marker=dict(color='rgba(0,0,0,0.4)', size=6)
            #             ))
            #         fig.add_trace(go.Histogram(y=y,
            #                 xaxis = 'x2',
            #                 marker = dict(
            #                     color = 'rgba(0,0,0,1)'
            #                 )
            #             ))
            #         fig.add_trace(go.Histogram(x=x,
            #                 yaxis = 'y2',
            #                 marker = dict(
            #                     color = 'rgba(0,0,0,1)'
            #                 )
            #             ))

            #         title = "Eigenvalues in <b>{}</b> dataset with ".format(dataset_name)
            #         title+="<b>k_opt={}; gamma={:.2E}; sigma={:.2E}</b>".format(comb[0], comb[1], comb[2])
            #         title+= " [<b>{} instances</b>]".format(freq)
            #         fig.update_layout(
            #             title = title,
            #             xaxis = dict(zeroline=False, showgrid=False,
            #                 domain = [0,0.85],
            #                 title='Real part'
            #             ),
            #             yaxis = dict(zeroline = False,
            #                 domain = [0,0.85],
            #                 title='Imaginary part'
            #             ),
            #             xaxis2 = dict(zeroline = False, showgrid = False,
            #                 domain = [0.85,1]
            #             ),
            #             yaxis2 = dict(zeroline = False, showgrid = False,
            #                 domain = [0.85,1]
            #             ),
            #             bargap = 0,
            #             hovermode = 'closest',
            #             showlegend = False
            #         )

            #         fig.show()

            print("case: ", end='')
            print("k_opt={}; gamma={:.2E}; sigma={:.2E}".format(int(comb[0]), comb[1], comb[2]), end=' ')
            print("[{:}]".format(str(freq) + " instances"))
            print("dtypes = {}".format(np.unique(dtypes)))
            print("Worst conditioning: {:.2E}".format(cond_worst))
            print("-" * 55)

            print("\n")

        display(HTML('<hr>'))
    #     print("\n"+"#"*100+"\n")