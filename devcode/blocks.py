import pandas as pd

from devcode.analysis.results import results_per_dataset
from devcode.simulation.settings import get_default_lssvm_gs_hyperparams, get_som_params
from devcode.utils import load_csv_as_pandas


all_metrics_header = ['Mean', 'Std. Deviation', 'Sens.', 'Spec.', 'f1 Score']


class ResultHandler:
    @staticmethod
    def global_ols_summary(datasets):
        global_ols_results_file = "results/global-results/ols"
        df_results = load_csv_as_pandas(global_ols_results_file)

        results_per_dataset(datasets, df_results, header=['Mean', 'Std. Deviation', 'Sens.', 'Spec.', 'f1 Score'],
                            only_accuracy=False)

    @staticmethod
    def global_lssvm_summary(datasets):
        df_results = pd.read_csv('results/global-results/lssvm/G-LSSVM - n_init=50 - 2019-08-28 (random states).csv')
        # df_results = load_csv_as_pandas(global_ols_results_file)
        lssvm_params = get_default_lssvm_gs_hyperparams()
        # exp_params = lssvm_params, params_keys = ['gamma', 'sigma']
        results_per_dataset(datasets, df_results, header=all_metrics_header, only_accuracy=False)

    @staticmethod
    def local_lssvm_summary(datasets):
        df_results = load_csv_as_pandas('results/local-results/cbic/temp_llssvm_cbic/results')
        # exp_params = lssvm_params, params_keys = ['gamma', 'sigma']
        results_per_dataset(datasets, df_results, header=['Mean', 'Std. Deviation', 'Sens.', 'Spec.', 'f1 Score'],
                            only_accuracy=False)

    @staticmethod
    def regional_ols_summary(datasets):
        # Hyperparameters grid search:
        alphas = [0.1, 0.3, 0.5]
        sigmas = [3.0, 6.5, 10.0]
        epochs = [100]

        df_results = pd.read_csv('results/regional-results/ROLS - all - n_res=100 - 2019-07-10.csv')

        som_params = get_som_params(alphas, sigmas, epochs)

        header = list(som_params[0].keys()) + ['Mean', 'Std. Deviation']  # 'Minimum', 'Maximum', 'Median',

        results_per_dataset(datasets, df_results, header, exp_params=som_params,
                            params_keys=['alpha0', 'sigma0', 'nEpochs'])

    @staticmethod
    def regional_lssvm_summary(datasets):
        df_results = load_csv_as_pandas('results/regional-results/temp_rlssvm/results')

        lssvm_params = get_default_lssvm_gs_hyperparams()

        #                             params_keys=['gamma', 'sigma']
        params_keys = ['gamma', 'sigma']
        header      = params_keys + ['Mean', 'Std. Deviation']  # 'Minimum', 'Maximum', 'Median',

        # results_per_dataset(datasets, df_results, header, exp_params=lssvm_params, params_keys=params_keys)
        results_per_dataset(datasets, df_results, header=all_metrics_header, only_accuracy=False)
