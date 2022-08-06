import pandas as pd
import numpy as np

from devcode.utils import load_csv_as_pandas
from devcode.utils.evaluation import cm2f1, cm2acc, cm2sen, cm2esp

df = {
    'global'    : load_csv_as_pandas(path="results/local-results/cbic/temp_glssvm_cbic"),
    'local'     : load_csv_as_pandas(path="results/local-results/cbic/temp_llssvm_cbic/results"),
    'regional'  : load_csv_as_pandas(path="results/regional-results/temp_rlssvm_somfix/results")
}

datasets   = np.unique(df['global']['dataset_name'].values).tolist()
df_results = pd.read_csv("results/regional-results/ROLS - all - n_res=100 - 2019-07-10.csv")

num = 3

alphas = np.linspace(0.1, 0.5, num=num).tolist()
sigmas = np.linspace(3, 10, num=num).tolist()
epochs = np.linspace(100, 500, num=num, dtype='int').tolist()

som_params = [
    {
        "alpha0": alpha0
        , "sigma0": sigma0
        , "nEpochs": nEpochs
    }
    for alpha0 in alphas
    for sigma0 in sigmas
    for nEpochs in epochs
]

# header = list(som_params[0].keys()) + ['Minimum', 'Maximum', 'Median', 'Mean', 'Std. Deviation']
header = list(som_params[0].keys()) + \
         ["acc2filter", "Accuracy", "Sens.", "Spec.", "F1"]

df_ds = {dataset_name: {'tr': None, 'ts': None} for dataset_name in datasets}

for dataset_name in datasets:  # For this specific dataset
    print(dataset_name)
    df_dataset = df_results.loc[df_results['dataset_name'] == dataset_name]  # get simulation results

    # matriz que guardará resultados numéricos
    df_data = {
        'tr': np.empty((len(som_params), len(header)), dtype=object),
        'ts': np.empty((len(som_params), len(header)), dtype=object)
    }
    for set_ in ['tr', 'ts']:
        print(set_)
        count = 0
        for params in som_params:
            df_case = df_dataset.loc[(df_dataset['alpha0'] == params['alpha0']) &
                             (df_dataset['sigma0'] == params['sigma0']) &
                             (df_dataset['nEpochs'] == params['nEpochs'])]

            # converting confusion matrix from string to numpy array
            cm = np.array(
                [
                    [int(x) for x in result[1:-1].split()]
                    for result in df_case[f'cm_{set_}'].values
                ]
            )

            if len(cm.shape) < 2:
                #                 display(cm)
                #                 display(cm.shape)
                df_case
                continue

            length = cm.shape[1]
            cm_side = int(np.sqrt(length))

            acc = [0] * len(cm)
            sens = [0] * len(cm)
            spec = [0] * len(cm)
            f1 = [0] * len(cm)
            for i in range(len(cm)):
                cm_temp = np.reshape(cm[i], (cm_side, cm_side))

                acc[i] = cm2acc(cm_temp) * 100
                sens[i] = cm2sen(cm_temp) * 100
                spec[i] = cm2esp(cm_temp) * 100
                f1[i] = cm2f1(cm_temp) * 100

            df_data[set_][count, :] = np.matrix([
                params['alpha0'], params['sigma0'], params['nEpochs'], np.mean(acc),
                "{:.2f} \$\pm\$ {:.2f}".format(np.mean(acc), np.std(acc)),
                "{:.2f}".format(np.mean(sens)),
                "{:.2f}".format(np.mean(spec)),
                "{:.2f}".format(np.mean(f1))
            ], dtype=object)

            count += 1

        df_ds[dataset_name][set_] = pd.DataFrame(df_data[set_], columns=header)
        # display(df_ds[dataset_name][set_].head())
    print('-' * 100, '\n' * 2)

from IPython.core.display import display, HTML
import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)  # enabling plot within jupyter notebook

set_dict = {'treino': 'cm_tr', 'teste': 'cm_ts'}

for dataset_name in datasets:
    display(HTML('<center><h1>' + dataset_name + '</h1></center>'))

    data = {}
    for classifier in df.keys():
        df_dataset = df[classifier].loc[df[classifier]['dataset_name'] == dataset_name]

        n_exp = len(df_dataset)
        data[classifier] = {
            'treino': [None] * n_exp,
            'teste': [None] * n_exp
        }

        for set_ in set_dict:
            cm_series = df_dataset[set_dict[set_]].values

            for i in range(n_exp):
                temp_cm = np.frombuffer(eval(cm_series[i]), dtype='int64')
                cm = temp_cm.reshape(int(len(temp_cm) ** (1 / 2)), -1)

                # acuracia:
                data[classifier][set_][i] = cm2acc(cm) * 100

    boxs = []
    for set_ in set_dict:
        for classifier in df.keys():
            cor = {
                'global': "rgba(44, 160, 101, 0.5)",
                'local': "rgba(93, 164, 214, 0.5)",
                'regional': "rgba(155, 89, 182,1.0)"
            }
            boxs.append(
                go.Box(
                    y=data[classifier][set_],
                    x=["{}{} {}-LSSVM".format(set_[0].upper(), set_[1:],
                                              classifier[0].upper())
                       ] * len(data[classifier][set_]),
                    name="{}-LSSVM".format(classifier[0].upper()),
                    boxmean='sd',
                    marker_color=cor[classifier],
                    #                     showlegend = False if set_=='treino' else True
                )
            )

    layout = go.Layout(
        #         title = "Acurácia nos conjuntos de treino e teste [<b>{}</b>]".format(dataset_name),
        yaxis=dict(title="Acurácia (%)"),
        showlegend=False,
        legend=dict(x=.875, y=1)
    )

    fig = go.Figure(data=boxs, layout=layout)
    width = 2
    fig.add_trace(go.Scatter(
        x=['Treino G-LSSVM', 'Treino L-LSSVM', 'Treino R-LSSVM', 'Treino G-LSSVM'],
        y=[np.mean(data['global']['treino']), np.mean(data['local']['treino']),
           np.mean(data['regional']['treino']), np.mean(data['global']['treino'])],
        mode='lines+markers',
        line=dict(
            color="RoyalBlue",
            dash="dashdot",
            #             dash="dot",
            width=width
        ),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=['Teste G-LSSVM', 'Teste L-LSSVM', 'Teste R-LSSVM', 'Teste G-LSSVM'],
        y=[np.mean(data['global']['teste']), np.mean(data['local']['teste']),
           np.mean(data['regional']['teste']), np.mean(data['global']['teste'])],
        mode='lines+markers',
        line=dict(
            color="RoyalBlue",
            dash="dashdot",
            #             dash="dot",
            width=width
        ),
        showlegend=False
    ))

    fig.update_layout(
        margin=dict(l=20, r=5, t=5, b=20),
        #         paper_bgcolor="LightSteelBlue",
    )

    fig.show()

    fig.write_image("results/regional-results/temp-images/r-lssvm_{}.pdf".format(dataset_name))

    display(HTML('<hr>'))