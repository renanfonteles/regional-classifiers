import numpy as np
import pandas as pd

from functools import partial
from copy import copy

from multiprocessing import Pool

from devcode.settings import get_default_lssvm_gs_hyperparams
from devcode.utils.simulation import evalGOLS, eval_GLSSVM
from devcode.utils import initialize_file

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    from load_dataset import datasets

    datasets_names = ['wf2f', 'wf4f', 'wf24f'] # 'pk', 'vc2c', 'vc3c',

    # Constant hyperparameters:
    test_size  = 0.5
    scale_type = 'min-max'
    n_init     = 50         # number of independent runs

    hps_cases = get_default_lssvm_gs_hyperparams()

    # Load vector of random states for train/test split
    rnd_states_file = "results/local-results/G-LSSVM - n_init=50 - 2019-08-28 (random states).csv"
    random_states   = np.unique(pd.read_csv(rnd_states_file, usecols=['random_state']).values).tolist()

    # random_states = np.random.randint(np.iinfo(np.int32).max, size=n_init).tolist()
    cases = [
        {
             "dataset_name": dataset_name
            ,"random_state": random_state
        }
        # hyperparameters possible values
        for dataset_name in datasets_names
        for random_state in random_states
    ]

    print(" ")
    print("# of data set runs = {}".format(len(cases)))

    global_results_file = f"results/simulations"
    header = ["dataset_name", "random_state", "$\gamma$", "$\sigma$", "eigenvalues", "eigenvalues_dtype",
              "cm_tr", "cm_ts"]
    filename = f"results/simulations/GLSSVM - all - n_res={n_init}.csv"

    simulation_file = initialize_file(filename, header)

    pool = Pool()
    data = pool.map(partial(eval_GLSSVM, datasets, simulation_file, header, scale_type, test_size, hps_cases), cases)
    pool.close()
    pool.join()
