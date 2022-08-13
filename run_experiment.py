from devcode.simulation.settings import ExperimentHandler

if __name__ == '__main__':
    from load_dataset import get_datasets
    datasets      = get_datasets()
    dataset_names = ["pk", "vc2c", "vc3c", "wf2f", "wf4f", "wf24f"]

    ExperimentHandler.run_experiment(datasets, dataset_names, n_samples=5, test_size=0.5, random_states=None)


