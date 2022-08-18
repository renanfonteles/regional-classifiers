import numpy as np
import tqdm

from devcode.analysis.clustering import cluster_val_metrics, get_header_optimal_k_lssvm_hps, \
    regional_cluster_val_metrics
from devcode.simulation.settings import get_default_lssvm_gs_hyperparams, default_regional_cases

from devcode.utils import initialize_file
from devcode.utils.simulation import eval_GLSSVM, evalRLM, eval_LLSSVM, eval_RLSSVM, set_per_round

from multiprocessing import Pool
from functools import partial


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


class ExperimentHandler:

    @classmethod
    def run_experiment(cls, datasets, dataset_names, n_samples=50, scale_type='min-max', test_size=0.5,
                       random_states=None, random_generator=None, multiprocessing=False):
        """
            Experiment Part 2: global vs local vs regional comparing using LSSVM as base classifier
        """

        if random_states is None:
            random_states = ExperimentSettings.get_random_states(n_samples=n_samples,
                                                                 random_generator=random_generator)

        if multiprocessing:
            cls._run_multiprocessing(datasets, dataset_names, scale_type=scale_type, test_size=test_size,
                                     random_states=random_states)
        else:
            cls._run_sequentially(datasets, dataset_names, scale_type=scale_type, test_size=test_size,
                                  random_states=random_states)

    @classmethod
    def _run_partial_sequentially(cls, case_func, run_func, datasets, dataset_names, scale_type, test_size,
                                  random_states):
        sim_file, header, cases, hps_cases = case_func(dataset_names, random_states=random_states)

        # run_sets = set_per_round(cases, datasets, test_size, scale_type)

        n_cases = len(cases)
        for i in range(n_cases):
            case = cases[i]
            print(f"Case #{i}: {case}")
            run_func(datasets, sim_file, header, scale_type, test_size, hps_cases, case)

    @classmethod
    def _run_sequentially(cls, datasets, dataset_names, scale_type, test_size, random_states):
        # Global LSSVM
        cls._run_partial_sequentially(ExperimentSettings.global_lssvm_settings, eval_GLSSVM, datasets,
                                      dataset_names, scale_type, test_size, random_states)

        # Local LSSVM
        cls._run_partial_sequentially(ExperimentSettings.local_regional_lssvm_cases, eval_LLSSVM, datasets,
                                      dataset_names, scale_type, test_size, random_states)

        # Regional LSSVM
        cls._run_partial_sequentially(ExperimentSettings.regional_lssvm_cases, eval_RLSSVM, datasets,
                                      dataset_names, scale_type, test_size, random_states)

    @classmethod
    def _run_partial_multiprocessing(cls, pool, case_func, run_func, datasets, dataset_names, scale_type, test_size,
                                     random_states):
        sim_file, header, cases, hps_cases = case_func(dataset_names, random_states=random_states)

        data_model = pool.map(partial(run_func, datasets, sim_file, header, scale_type, test_size, hps_cases), cases)

        return data_model

    @classmethod
    def _run_multiprocessing(cls, datasets, dataset_names, scale_type, test_size, random_states):
        pool = Pool()

        # Global LSSVM
        data_global_lssvm = cls._run_partial_multiprocessing(
            pool, ExperimentSettings.global_lssvm_settings, eval_GLSSVM, datasets, dataset_names, scale_type,
            test_size, random_states)

        # Local LSSVM
        data_local_lssvm = cls._run_partial_multiprocessing(
            pool, ExperimentSettings.local_regional_lssvm_cases, eval_LLSSVM, datasets, dataset_names, scale_type,
            test_size, random_states)

        # Regional LSSVM
        data_regional_lssvm = cls._run_partial_multiprocessing(
            pool, ExperimentSettings.regional_lssvm_cases, eval_RLSSVM, datasets, dataset_names, scale_type,
            test_size, random_states)

        pool.close()
        pool.join()
