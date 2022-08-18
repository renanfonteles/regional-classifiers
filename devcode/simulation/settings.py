import numpy as np

from devcode.analysis.clustering import regional_cluster_val_metrics, cluster_val_metrics, \
    get_header_optimal_k_lssvm_hps
from devcode.utils import initialize_file


def default_regional_cases(datasets_names, random_states):
    alphas = default_alphas()
    sigmas = default_sigmas()
    epochs = default_epochs()

    alpha0, sigma0 , nEpochs = 0.1, 10, 300

    cases = [{"dataset_name": dataset_name, "random_state": random_state,
              "som_params": {"alpha0": alpha0, "sigma0": sigma0, "nEpochs": nEpochs}}
             for dataset_name in datasets_names
             for random_state in random_states
             # for alpha0 in alphas
             # for sigma0 in sigmas
             # for nEpochs in epochs
             ]

    return cases


def get_default_lssvm_gs_hyperparams():
    # Hyperparameters grid search:
    gammas = np.logspace(-6.0, 6.0, num=7).tolist()
    sigmas = np.logspace(-0.5, 3.0, num=5).tolist()

    print("gammas = {}".format(gammas))
    print("sigmas = {}".format(sigmas))

    hps_cases = [{"gamma": gamma, "sigma": sigma}
                 for gamma in gammas
                 for sigma in sigmas]

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
    som_params = [{"alpha0": alpha0, "sigma0": sigma0, "nEpochs": nEpochs}
                  for alpha0 in alphas
                  for sigma0 in sigmas
                  for nEpochs in epochs]

    return som_params


def default_alphas(num=3, alphas=(0.1, 0.3, 0.5)):
    return alphas if alphas else np.linspace(0.1, 0.5,  num=num).tolist()


def default_sigmas(num=3, sigmas=(3.0, 6.5, 10.0)):
    return sigmas if sigmas else np.linspace(3, 10, num=num).tolist()


def default_epochs(num=3, epochs=(100,)):
    return epochs if epochs else np.linspace(100,  500, num=num, dtype='int').tolist()


class ExperimentSettings:
    @staticmethod
    def get_random_states(n_samples, random_generator=None):
        # Vector of random states for train/test split
        if random_generator:
            max_int = np.iinfo(np.int32).max
            random_states = random_generator.integers(max_int, size=n_samples).tolist()
        else:
            random_states = np.random.randint(np.iinfo(np.int32).max, size=n_samples).tolist()
        return random_states

    @classmethod
    def _define_header(cls, additional_header):
        additional_header = additional_header if additional_header else []
        header_head = ["dataset_name", "random_state"]
        header_tail = ["eigenvalues", "eigenvalues_dtype", "cm_tr", "cm_ts"]

        return header_head + additional_header + header_tail

    @classmethod
    def _default_cases(cls, datasets_names, random_states):
        cases = [{"dataset_name": dataset_name, "random_state": random_state}
                 for dataset_name in datasets_names
                 for random_state in random_states]

        return cases

    @classmethod
    def _create_simulation_settings(cls, sim_name, datasets_names, random_states, cases=None,
                                    additional_header=None, base_path="results/"):
        n_samples = len(random_states)

        cases  = cases if cases else cls._default_cases(datasets_names, random_states)
        header = cls._define_header(additional_header)

        filename = f"{base_path}{sim_name} - all - n_res={n_samples}.csv"
        simulation_file = initialize_file(filename, header)

        return simulation_file, header, cases

    @classmethod
    def default_settings(cls, model_name, datasets_names, random_states):
        simulation_file, header, cases = cls._create_simulation_settings(
            sim_name=model_name, datasets_names=datasets_names, random_states=random_states)

        return simulation_file, header, cases, None

    @classmethod
    def global_ols_settings(cls, datasets_names, random_states):
        return cls.default_settings("G-OLS", datasets_names, random_states)

    @classmethod
    def regional_ols_settings(cls, datasets_names, random_states):
        return cls.default_settings("R-OLS", datasets_names, random_states)

    @classmethod
    def global_lssvm_settings(cls, datasets_names, random_states):
        hps_cases = get_default_lssvm_gs_hyperparams()
        simulation_file, header, cases = cls._create_simulation_settings(
            sim_name="GLSSVM", datasets_names=datasets_names, random_states=random_states,
            additional_header=["$\gamma$", "$\sigma$"])

        return simulation_file, header, cases, hps_cases

    @classmethod
    def regional_lssvm_cases(cls, datasets_names, random_states):
        return cls.local_regional_lssvm_cases(datasets_names, random_states, is_regional=True)

    @classmethod
    def local_regional_lssvm_cases(cls, datasets_names, random_states, is_regional=False):
        hps_cases = get_default_lssvm_gs_hyperparams()

        if is_regional:
            sim_name        = "RLSSVM"
            cluster_metrics = regional_cluster_val_metrics
            cases           = default_regional_cases(datasets_names, random_states)
        else:
            sim_name        = "LLSSVM"
            cluster_metrics = cluster_val_metrics
            cases           = cls._default_cases(datasets_names, random_states)

        optimal_k_hps = get_header_optimal_k_lssvm_hps(cluster_metrics=cluster_metrics)

        additional_header = ["# empty regions", "# homogeneous regions", "$\gamma_{opt}$ [CV]", "$\sigma_{opt}$ [CV]"]

        additional_header += optimal_k_hps
        additional_header += ["$k_{opt}$ [CV]"]
        additional_header += ['$k_{opt}$ ' + '[{}]'.format(metric['name']) for metric in cluster_metrics]
        additional_header += ['cv_score [{}]'.format(metric['name']) for metric in cluster_metrics]

        simulation_file, header, cases = cls._create_simulation_settings(
            sim_name=sim_name, datasets_names=datasets_names, random_states=random_states, cases=cases,
            additional_header=additional_header)

        return simulation_file, header, cases, hps_cases
