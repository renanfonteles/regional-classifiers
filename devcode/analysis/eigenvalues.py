import numpy as np
import pandas as pd
import itertools


from IPython.core.display import display, HTML


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


def eigenvalues_analysis_regional(datasets, df_model):
    for dataset_name in datasets:
        display(HTML('<center><h1>' + dataset_name + '</h1></center>'))
        print(" ")

        # get dataframe of the specific dataset
        df_dataset = df_model.loc[df_model['dataset_name'] == dataset_name]

        print("{}: {} runs. ".format(dataset_name, len(df_dataset)))

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