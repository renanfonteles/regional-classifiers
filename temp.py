import pandas as pd
import numpy as np
import plotly.offline as py

from devcode.analysis.results import results_per_dataset
from devcode.simulation import ResultProcessor
from devcode.utils import load_csv_as_pandas

_dataframes_dict = {
    'global'    : load_csv_as_pandas(path="results/local-results/cbic/temp_glssvm_cbic"),
    'local'     : load_csv_as_pandas(path="results/local-results/cbic/temp_llssvm_cbic/results"),
    'regional'  : load_csv_as_pandas(path="results/regional-results/temp_rlssvm_somfix/results")
}

ds_names     = ['pk', 'vc2c', 'vc3c', 'wf2f', 'wf4f', 'wf24f']

py.init_notebook_mode(connected=True)  # enabling plot within jupyter notebook

set_dict = {'treino': 'cm_tr', 'teste': 'cm_ts'}

model_labels = _dataframes_dict.keys()


ResultProcessor.process_results_in_multiple_datasets(ds_names, _dataframes_dict,
                                                     ResultProcessor.regional_k_optimal_histogram)
