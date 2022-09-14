import numpy as np

from devcode import PRE_PROCESS_PATH
from devcode.simulation.pipeline import ExperimentHandler
from devcode.utils import FileUtils

if __name__ == '__main__':
    """
        Run pipeline simulation to generate part2 results in manuscript.
        
        Note: Set the devcode.GLOBAL_BASE_PATH for selecting the directory where the results will be saved.
    """
    import warnings
    warnings.filterwarnings('ignore')

    from load_dataset import get_datasets
    datasets      = get_datasets()
    dataset_names = ["pk", "vc2c", "vc3c", "wf2f", "wf4f", "wf24f"]

    rng = np.random.default_rng(seed=42)
    random_states = FileUtils.load_pickle_file(file_path=f"{PRE_PROCESS_PATH}/random_states.pickel")

    ExperimentHandler.run_experiment_part2(datasets, dataset_names, n_samples=50, test_size=0.5,
                                           random_states=random_states, multiprocessing=False)
