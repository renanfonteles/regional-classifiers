import numpy as np

from devcode.simulation.pipeline import ExperimentHandler
from devcode.utils import FileUtils

glssvm_result_path = "results/local-results/cbic/temp_glssvm_cbic"
llssvm_result_path = "results/local-results/cbic/temp_llssvm_cbic/results"
rlssvm_result_path = "results/regional-results/temp_rlssvm_somfix/results"


def get_random_states(file_names):
    import re
    all_rnd_state = [int(re.split(" ", file_name)[-1][:-5]) for file_name in file_names]
    return list(set(all_rnd_state))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    from load_dataset import get_datasets
    datasets      = get_datasets()
    dataset_names = ["pk", "vc2c", "vc3c", "wf2f", "wf4f", "wf24f"]

    rng = np.random.default_rng(seed=42)
    random_states = get_random_states(FileUtils.get_file_names_in_dir(dir_path=glssvm_result_path))

    ExperimentHandler.run_experiment_part1(datasets, dataset_names, n_samples=50, test_size=0.5,
                                           random_states=random_states, multiprocessing=False)

    # ExperimentHandler.run_experiment_part2(datasets, dataset_names, n_samples=50, test_size=0.5,
    #                                        random_states=random_states, multiprocessing=False)

