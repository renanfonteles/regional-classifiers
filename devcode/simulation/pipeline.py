from devcode.simulation.settings import ExperimentSettings
from devcode.simulation.fit_evaluation import eval_GLSSVM, eval_LLSSVM, eval_RLSSVM, eval_ROLS, eval_GOLS

from multiprocessing import Pool
from functools import partial


class ExperimentHandler:

    @classmethod
    def run_experiment_part2(cls, datasets, dataset_names, n_samples=50, scale_type='min-max', test_size=0.5,
                             random_states=None, random_generator=None, multiprocessing=False):
        """
            Experiment Part 2: global vs local vs regional comparing using LSSVM as base classifier
        """

        experiment_settings = [
            (ExperimentSettings.global_lssvm_settings,      eval_GLSSVM),
            (ExperimentSettings.local_lssvm_settings,       eval_LLSSVM),
            (ExperimentSettings.regional_lssvm_settings,    eval_RLSSVM)

        ]

        cls.run_experiment(experiment_settings, datasets, dataset_names, n_samples=n_samples, scale_type=scale_type,
                           test_size=test_size, random_states=random_states, random_generator=random_generator,
                           multiprocessing=multiprocessing)

    @classmethod
    def run_experiment(cls, experiment_settings, datasets, dataset_names, n_samples=50, scale_type='min-max',
                       test_size=0.5, random_states=None, random_generator=None, multiprocessing=False):
        """
            Run experiment based on specific setting and evaluation functions
        """

        if random_states is None:
            random_states = ExperimentSettings.get_random_states(n_samples=n_samples,
                                                                 random_generator=random_generator)

        if multiprocessing:
            cls._run_multiprocessing(experiment_settings, datasets, dataset_names, scale_type=scale_type,
                                     test_size=test_size, random_states=random_states)
        else:
            cls._run_sequentially(experiment_settings, datasets, dataset_names, scale_type=scale_type,
                                  test_size=test_size, random_states=random_states)

    @classmethod
    def _run_partial_sequentially(cls, settings_func, run_func, datasets, dataset_names, scale_type, test_size,
                                  random_states):
        sim_file, header, cases, hps_cases = settings_func(dataset_names, random_states=random_states)

        n_cases = len(cases)
        for i in range(n_cases):
            case = cases[i]
            print(f"Case #{i + 1}: {case}")
            run_func(datasets, sim_file, header, scale_type, test_size, hps_cases, case)

    @classmethod
    def _run_sequentially(cls, experiment_settings, datasets, dataset_names, scale_type, test_size, random_states):

        for settings_func, eval_func in experiment_settings:
            cls._run_partial_sequentially(settings_func, eval_func, datasets, dataset_names, scale_type, test_size,
                                          random_states)

    @classmethod
    def _run_partial_multiprocessing(cls, pool, settings_func, run_func, datasets, dataset_names, scale_type, test_size,
                                     random_states):
        sim_file, header, cases, hps_cases = settings_func(dataset_names, random_states=random_states)

        data_model = pool.map(partial(run_func, datasets, sim_file, header, scale_type, test_size, hps_cases), cases)

        return data_model

    @classmethod
    def _run_multiprocessing(cls, experiment_settings, datasets, dataset_names, scale_type, test_size, random_states):
        pool = Pool()

        for settings_func, eval_func in experiment_settings:
            _ = cls._run_partial_multiprocessing(pool, settings_func, eval_func, datasets, dataset_names, scale_type,
                                                 test_size, random_states)

        pool.close()
        pool.join()
