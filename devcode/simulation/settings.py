import numpy as np
import tqdm

from devcode.analysis.clustering import cluster_val_metrics
from devcode.utils import initialize_file
from devcode.utils.simulation import eval_GLSSVM, evalRLM, eval_LLSSVM

from multiprocessing import Pool
from functools import partial

from devcode.utils.simulation import evalRLM
from devcode.utils import initialize_file


class ExperimentHandler:
    @staticmethod
    def get_random_states(n_samples):
        # vector of random states for train/test split
        random_states = np.random.randint(np.iinfo(np.int32).max, size=n_samples).tolist()
        return random_states

    @staticmethod
    def _global_lssvm_cases(datasets_names, random_states):
        n_samples = len(random_states)
        hps_cases = get_default_lssvm_gs_hyperparams()

        cases = [{"dataset_name": dataset_name, "random_state": random_state}
                 for dataset_name in datasets_names
                 for random_state in random_states]

        header = ["dataset_name", "random_state", "$\gamma$", "$\sigma$", "eigenvalues", "eigenvalues_dtype",
                  "cm_tr", "cm_ts"]

        filename = f"results/GLSSVM - all - n_res={n_samples}.csv"

        simulation_file = initialize_file(filename, header)

        return simulation_file, header, cases, hps_cases

    @staticmethod
    def _local_lssvm_cases(datasets_names, random_states):
        def get_optimal_k_hps():
            temp = [' '] * 2 * len(cluster_val_metrics)
            count = 0

            for metric in cluster_val_metrics:
                temp[count] = "$\gamma_{opt}$ " + "[{}]".format(metric['name'])
                temp[count + 1] = "$\sigma_{opt}$ " + "[{}]".format(metric['name'])
                count += 2

            return temp

        n_samples = len(random_states)
        hps_cases = get_default_lssvm_gs_hyperparams()

        alphas = [0.1, 0.3, 0.5]        # np.linspace(0.1, 0.5,  num=num).tolist()
        sigmas = [3.0, 6.5, 10.0]       # np.linspace(3,    10,   num=num).tolist()
        epochs = [100]                  # np.linspace(100,  500, num=num, dtype='int').tolist()

        cases = [{"dataset_name": dataset_name, "random_state": random_state,
                  "som_params": {"alpha0": alpha0, "sigma0": sigma0, "nEpochs": nEpochs}}
                 for dataset_name in datasets_names
                 for random_state in random_states
                 for alpha0 in alphas
                 for sigma0 in sigmas
                 for nEpochs in epochs
        ]

        optimal_k_hps = get_optimal_k_hps()

        header = ["dataset_name", "random_state", "# empty regions", "# homogeneous regions"] + \
                 ["$\gamma_{opt}$ [CV]", "$\sigma_{opt}$ [CV]"] + \
                 optimal_k_hps + \
                 ["$k_{opt}$ [CV]"] + \
                 ['$k_{opt}$ ' + '[{}]'.format(metric['name']) for metric in cluster_val_metrics] + \
                 ['cv_score [{}]'.format(metric['name']) for metric in cluster_val_metrics] + \
                 ["eigenvalues", "eigenvalues_dtype", "cm_tr", "cm_ts"]

        filename = f"results/LLSSVM - all - n_res={n_samples}.csv"

        simulation_file = initialize_file(filename, header)

        return simulation_file, header, cases, hps_cases

    @classmethod
    def run_experiment(cls, datasets, dataset_names, n_samples=20, scale_type='min-max', test_size=0.5,
                       random_states=None):

        if random_states is None:
            random_states = ExperimentHandler.get_random_states(n_samples=n_samples)

        pool = Pool()

        glssvm_sim_file, glssvm_header, glssvm_cases, lssvm_hps_cases = cls._global_lssvm_cases(
            dataset_names, random_states=random_states)
        llssvm_sim_file, llssvm_header, llssvm_cases, _ = cls._local_lssvm_cases(
            dataset_names, random_states=random_states)

        data_global = pool.map(
            partial(eval_GLSSVM, datasets, glssvm_sim_file, glssvm_header, scale_type, test_size, lssvm_hps_cases),
            glssvm_cases)

        data_local = pool.map(
            partial(eval_LLSSVM, datasets, llssvm_sim_file, llssvm_header, scale_type, test_size, lssvm_hps_cases),
            llssvm_cases)

        pool.close()
        pool.join()


def get_default_lssvm_gs_hyperparams():
    # Hyperparameters grid search:
    gammas = np.logspace(-6.0, 6.0, num=7).tolist()
    sigmas = np.logspace(-0.5, 3.0, num=5).tolist()

    print("gammas = {}".format(gammas))
    print("sigmas = {}".format(sigmas))

    hps_cases = [
        {"gamma": gamma,
         "sigma": sigma
         }
        for gamma in gammas
        for sigma in sigmas
    ]

    print("# of hps_cases = {}".format(len(hps_cases)))

    return hps_cases


def get_default_som_cases(datasets, random_states, alphas, sigmas, epochs):
    cases = [{
            "dataset_name": dataset_name,
            "random_state": random_state,
            "som_params"  : {"alpha0": alpha0, "sigma0": sigma0, "nEpochs": nEpochs}
        }
        # hyperparameters possible values
        for dataset_name in datasets.keys()
        for random_state in random_states
        for alpha0 in alphas
        for sigma0 in sigmas
        for nEpochs in epochs
    ]

    return cases


def get_som_params(alphas, sigmas, epochs):
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

    return som_params


def default_alphas(num=3):
    return np.linspace(0.1, 0.5,  num=num).tolist()


def default_sigmas(num=3):
    return np.linspace(3,    10,   num=num).tolist()


def default_epochs(num=3):
    return np.linspace(100,  500, num=num, dtype='int').tolist()



