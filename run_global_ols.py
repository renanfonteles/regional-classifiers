from functools import partial

import numpy as np
import tqdm

from multiprocessing import Pool

from devcode.utils.simulation import evalGOLS
from devcode.utils import initialize_file

if __name__ == '__main__':
    from load_dataset import datasets

    test_size     = 0.2
    scale_type     = 'min-max'
    n_resamplings = 100

    # vector of random states for train/test split
    random_states = np.random.randint(np.iinfo(np.int32).max, size=n_resamplings).tolist()

    cases = [
        {
            "dataset_name": dataset_name
            , "random_state": random_state
        }
        for dataset_name in datasets.keys()
        for random_state in random_states
    ]

    header   = ["dataset_name", "random_state", "cm_tr", "cm_ts"]
    filename = f"results/simulations/GOLS - all - n_res={n_resamplings}.csv"

    simulation_file = initialize_file(filename, header)

    pool = Pool()
    data = pool.map(partial(evalGOLS, datasets, simulation_file, header, scale_type, test_size, None), cases)
    # data = [result for result in tqdm.tqdm(pool.imap_unordered(evalGOLS, cases), total=len(cases))]
    pool.close()
    pool.join()
