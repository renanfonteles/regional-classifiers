import numpy as np
import tqdm

from multiprocessing import Pool
from functools import partial

from devcode.utils import initialize_file

if __name__ == '__main__':
    from devcode.utils.simulation import evalRLM
    from load_dataset import datasets

    test_size     = 0.2
    scale_type    = 'min-max'
    n_resamplings = 100

    num = 3

    alphas = [0.1, 0.3, 0.5]     # np.linspace(0.1, 0.5,  num=num).tolist()
    sigmas = [3.0, 6.5, 10.0]    # np.linspace(3,    10,   num=num).tolist()
    epochs = [100]               # np.linspace(100,  500, num=num, dtype='int').tolist()

    # vector of random states for train/test split
    random_states = np.random.randint(np.iinfo(np.int32).max, size=n_resamplings).tolist()

    cases = [{"dataset_name": dataset_name, "random_state": random_state,
              "som_params": {"alpha0": alpha0, "sigma0": sigma0, "nEpochs": nEpochs}}
             for dataset_name in datasets.keys() for random_state in random_states
             for alpha0 in alphas for sigma0 in sigmas for nEpochs in epochs
    ]

    header   = ["dataset_name", "random_state", "alpha0", "sigma0", "nEpochs", "cm_tr", "cm_ts"]
    filename = f"results/simulations/ROLS - all - n_res={n_resamplings}.csv"

    simulation_file = initialize_file(filename, header)

    print(cases[0:2])
    pool = Pool()
    data = pool.map(partial(evalRLM, datasets, simulation_file, header, scale_type, test_size, None), cases)
    pool.close()
    pool.join()
