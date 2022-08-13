import plotly.offline as py
import plotly.graph_objs as go

from IPython.core.display import display, HTML

from devcode.simulation import ResultProcessor
from devcode.utils.visualization import create_multiple_boxplots, add_line, set_figure

import pandas as pd
import numpy as np

from devcode.utils import load_csv_as_pandas
from devcode.utils.evaluation import cm2f1, cm2acc, cm2sen, cm2esp

df_results = {
    'global'    : load_csv_as_pandas(path="results/local-results/cbic/temp_glssvm_cbic"),
    'local'     : load_csv_as_pandas(path="results/local-results/cbic/temp_llssvm_cbic/results"),
    'regional'  : load_csv_as_pandas(path="results/regional-results/temp_rlssvm_somfix/results")
}

datasets = np.unique(df_results['global']['dataset_name'].values).tolist()

py.init_notebook_mode(connected=True)  # enabling plot within jupyter notebook

set_dict = {'treino': 'cm_tr', 'teste': 'cm_ts'}

model_labels = df_results.keys()


def process_result_func(dataframes_dict, ds_name):
    ResultProcessor.compare_boxplot_per_set(dataframes_dict, ds_name)
    ResultProcessor.compare_local_regional_k_optimal(dataframes_dict, ds_name)


ResultProcessor.process_results_in_multiple_datasets(datasets, df_results, process_result_func)
