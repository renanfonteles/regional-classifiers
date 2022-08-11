import numpy as np


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


def get_random_states(n_samples):
    # vector of random states for train/test split
    random_states = np.random.randint(np.iinfo(np.int32).max, size=n_samples).tolist()
    return random_states
