import pandas as pd
import numpy as np
import plotly.offline as py

from devcode.simulation import ResultProcessor
from devcode.utils import load_csv_as_pandas

df_results = {
    'global'    : load_csv_as_pandas(path="results/local-results/cbic/temp_glssvm_cbic"),
    'local'     : load_csv_as_pandas(path="results/local-results/cbic/temp_llssvm_cbic/results"),
    'regional'  : load_csv_as_pandas(path="results/regional-results/temp_rlssvm_somfix/results")
}

ds_names     = ['pk', 'vc2c', 'vc3c', 'wf2f', 'wf4f', 'wf24f']

py.init_notebook_mode(connected=True)  # enabling plot within jupyter notebook

set_dict = {'treino': 'cm_tr', 'teste': 'cm_ts'}

model_labels = df_results.keys()


def process_result_func(dataframes_dict, ds_name):
    ResultProcessor.compare_boxplot_per_set(dataframes_dict, ds_name)
    ResultProcessor.compare_local_regional_k_optimal(dataframes_dict, ds_name)
    ResultProcessor.local_cluster_analysis(dataframes_dict, ds_name)


# ResultProcessor.process_results_in_multiple_datasets(ds_names, df_results, process_result_func)

# ResultProcessor.overall_regional_heatmap_cluster_analysis(df_results, ds_names)
ResultProcessor.overall_local_heatmap_cluster_analysis(df_results, ds_names)