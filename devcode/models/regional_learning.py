import datetime
import numpy as np

from sklearn import linear_model
from sklearn.cluster import KMeans
from copy import copy

from devcode.analysis.clustering import RegionalUtils
from devcode.models.local_learning import BiasModel


class RegionalModel:
    """
        Class of Regional Models.
    """

    def __init__(self, SOM_class, Model_class, Cluster_class=None):
        self.SOM        = SOM_class
        self.Cluster    = Cluster_class
        self.Model      = Model_class

        self.region_labels      = []
        self.regional_models    = []
        self.empty_regions      = []

        self.targets_dim_ = None

    def fit(self, X, Y, verboses=0, SOM_params=None, cluster_params=None, Model_params=None):
        self.targets_dim_ = Y.shape[1]  # dimension of target values

        # SOM training
        if SOM_params is not None:
            self.SOM.fit(X=X, **SOM_params)

        # Cluster training
        if cluster_params is not None:
            self.Cluster = KMeans(**cluster_params).fit(self.SOM.neurons)

        k_opt = cluster_params["n_clusters"]

        # Model training
        self.region_labels = self.regionalize(X)  # finding labels of datapoints
        if verboses == 1:
            print("Start of Model training at {}".format(datetime.datetime.now()))

        self.regional_models = [None] * k_opt
        for r in range(k_opt):  # for each region
            Xr = X[np.where(self.region_labels == r)[0]]
            Yr = Y[np.where(self.region_labels == r)[0]]

            model_r = None
            if len(Xr) == 0:  # empty region
                self.empty_regions.append(r)

            elif len(np.unique(Yr, axis=0)) == 1:  # homogenous region, only one class
                model_r = BiasModel(np.unique(Yr, axis=0))

            else:  # region not empty or homogeneous
                self.Model.fit(Xr, Yr)
                model_r = copy(self.Model)

            self.regional_models[r] = model_r

    def regionalize(self, X):
        regions = np.zeros(len(X), dtype='int')
        for i in range(len(X)):  # for each datapoint
            winner_som_idx, dist_matrix = self.SOM.get_winner(
                X[i], dim=1, dist_matrix=True)  # find closest neuron
            region = self.Cluster.labels_[winner_som_idx]  # find neuron label index in kmeans

            # if the region don't have a model is because it didn't have datapoints in the train set
            if region in self.empty_regions:
                dead_neurons, = np.where(self.Cluster.labels_ == region)
                dist_matrix[dead_neurons] = np.inf  # taking off dead neurons from the play

                temp = np.argmin(dist_matrix)
                region = self.Cluster.labels_[temp]

            regions[i] = region

        return regions

    def predict(self, X):
        try:
            n_samples = len(X)  # number of samples

            # Searching for a non-empty region
            regions = [i for i in range(self.Cluster.n_clusters)]
            not_empty_regions = list(set(regions) - set(self.empty_regions))

            #         temp = self.regional_models[not_empty_regions[0]].intercept_
            predictions = np.zeros((n_samples, self.targets_dim_))

            regions = self.regionalize(X)
            for i in range(n_samples):
                predictions[i, :] = self.regional_models[regions[i]].predict(X[i].reshape(1, -1))

            return predictions
        except AttributeError:
            print("Regional prediction error")
