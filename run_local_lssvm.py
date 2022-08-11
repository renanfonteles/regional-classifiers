import numpy as np

from devcode.settings import get_default_lssvm_gs_hyperparams
from devcode.utils import initialize_file
from devcode.utils.simulation import eval_LLSSVM

from multiprocessing import Pool
from functools import partial

if __name__ == '__main__':
    from devcode.analysis.clustering import cluster_val_metrics
    from load_dataset import datasets

    test_size     = 0.2
    scale_type    = 'min-max'
    n_resamplings = 100

    num = 3

    alphas = [0.1, 0.3, 0.5]     # np.linspace(0.1, 0.5,  num=num).tolist()
    sigmas = [3.0, 6.5, 10.0]    # np.linspace(3,    10,   num=num).tolist()
    epochs = [100]               # np.linspace(100,  500, num=num, dtype='int').tolist()

    hps_cases = get_default_lssvm_gs_hyperparams()

    random_states = np.random.randint(np.iinfo(np.int32).max, size=n_resamplings).tolist()

    cases = [{"dataset_name": dataset_name, "random_state": random_state,
              "som_params": {"alpha0": alpha0, "sigma0": sigma0, "nEpochs": nEpochs}}
             for dataset_name in datasets.keys() for random_state in random_states
             for alpha0 in alphas for sigma0 in sigmas for nEpochs in epochs
    ]

    temp = [' '] * 2 * len(cluster_val_metrics)
    count = 0

    for metric in cluster_val_metrics:
        temp[count] = "$\gamma_{opt}$ " + "[{}]".format(metric['name'])
        temp[count + 1] = "$\sigma_{opt}$ " + "[{}]".format(metric['name'])
        count += 2

    header = ["dataset_name", "random_state", "# empty regions", "# homogeneous regions"] + \
             ["$\gamma_{opt}$ [CV]", "$\sigma_{opt}$ [CV]"] + \
             temp + \
             ["$k_{opt}$ [CV]"] + \
             ['$k_{opt}$ ' + '[{}]'.format(metric['name']) for metric in cluster_val_metrics] + \
             ['cv_score [{}]'.format(metric['name']) for metric in cluster_val_metrics] + \
             ["eigenvalues", "eigenvalues_dtype", "cm_tr", "cm_ts"]

    filename = f"results/simulations/LLSSVM - all - n_res={n_resamplings}.csv"

    simulation_file = initialize_file(filename, header)

    pool = Pool()
    data = pool.map(partial(eval_LLSSVM, datasets, simulation_file, header, scale_type, test_size, hps_cases), cases)
    pool.close()
    pool.join()
