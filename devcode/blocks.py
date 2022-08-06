from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def data_preprocessing(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    temp   = scaler.fit_transform(data.data)

    pca = PCA(n_components=2)
    X   = pca.fit_transform(temp)

    return X


# def temp1():
#     # TODO: vou ter que separar os resultados por parâmetros do SOM porque
#     # fiz a otimização de hiper-parâmetros erroneamente
#     # hyperparameters grid search:
#     num = 3
#
#     alphas = np.linspace(0.1, 0.5, num=num).tolist()
#     sigmas = np.linspace(3, 10, num=num).tolist()
#     epochs = np.linspace(100, 500, num=num, dtype='int').tolist()
#
#     som_params = [
#         {
#             "alpha0": alpha0
#             , "sigma0": sigma0
#             , "nEpochs": nEpochs
#         }
#         for alpha0 in alphas
#         for sigma0 in sigmas
#         for nEpochs in epochs
#     ]
#
#     # header = list(som_params[0].keys()) + ['Minimum', 'Maximum', 'Median', 'Mean', 'Std. Deviation']
#     header = list(som_params[0].keys()) + ["acc2filter", "Accuracy", "Sens.", "Spec.", "F1"]
#
#     df_ds = {dataset_name: {'tr': None, 'ts': None} for dataset_name in datasets}
#
#     for dataset_name in datasets:  # For this specific dataset
#         print(dataset_name)
#         df = df_results.loc[df_results['dataset_name'] == dataset_name]  # Get simulation results
#
#         # Matrix that stores results
#         df_data = {
#             'tr': np.empty((len(som_params), len(header)), dtype=object),
#             'ts': np.empty((len(som_params), len(header)), dtype=object)
#         }
#         for set_ in ['tr', 'ts']:
#             print(set_)
#             count = 0
#             for params in som_params:
#                 df_case = df.loc[(df['alpha0'] == params['alpha0']) &
#                                  (df['sigma0'] == params['sigma0']) &
#                                  (df['nEpochs'] == params['nEpochs'])]
#
#                 # Converting confusion matrix from string to numpy array
#                 cm = np.array(
#                     [
#                         [int(x) for x in result[1:-1].split()]
#                         for result in df_case[f'cm_{set_}'].values
#                     ]
#                 )
#
#                 if len(cm.shape) < 2:
#                     df_case
#                     continue
#
#                 length = cm.shape[1]
#                 cm_side = int(np.sqrt(length))
#
#                 acc = [0] * len(cm)
#                 sens = [0] * len(cm)
#                 spec = [0] * len(cm)
#                 f1 = [0] * len(cm)
#
#                 for i in range(len(cm)):
#                     cm_temp = np.reshape(cm[i], (cm_side, cm_side))
#
#                     acc[i] = cm2acc(cm_temp) * 100
#                     sens[i] = cm2sen(cm_temp) * 100
#                     spec[i] = cm2esp(cm_temp) * 100
#                     f1[i] = cm2f1(cm_temp) * 100
#
#                 df_data[set_][count, :] = np.matrix([
#                     params['alpha0'], params['sigma0'], params['nEpochs'], np.mean(acc),
#                     "{:.2f} \$\pm\$ {:.2f}".format(np.mean(acc), np.std(acc)),
#                     "{:.2f}".format(np.mean(sens)),
#                     "{:.2f}".format(np.mean(spec)),
#                     "{:.2f}".format(np.mean(f1))
#                 ], dtype=object)
#
#                 count += 1
#
#             df_ds[dataset_name][set_] = pd.DataFrame(df_data[set_], columns=header)
#         print('-' * 100, '\n' * 2)


# def temp2():
#     for set_ in ['tr', 'ts']:
#         print(set_)
#         df_ds.keys()
#         df_temp = {}
#
#         for dataset_name, df_ts_ts in df_ds.items():
#             df_temp[dataset_name] = df_ts_ts[set_]
#
#         data = np.array([df.sort_values(
#             'acc2filter', ascending=False).iloc[0, :].values for df in df_temp.values()
#                          ])
#         idx_label = list(df_ds.keys())
#         df_rols = pd.DataFrame(data, columns=header, index=[idx_label])
#         display(df_rols)