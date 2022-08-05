import datetime
import numpy as np

from sklearn import linear_model
from sklearn.cluster import KMeans
from copy import copy

from local_learning import BiasModel


class RegionalModel:
    'Class of Regional Models.'

    def __init__(self, SOM_class, Model_class, Cluster_class=None):
        self.SOM = SOM_class
        self.Cluster = Cluster_class
        self.Model = Model_class
        self.region_labels = []
        self.regional_models = []
        self.empty_regions = []
        self.targets_dim_ = None

    def fit(self, X, Y, verboses=0, SOM_params=None, Cluster_params=None, Model_params=None):
        self.targets_dim_ = Y.shape[1]  # dimension of target values

        # SOM training
        if SOM_params is not None:
            if verboses == 1: print("Start of SOM training at {}".format(datetime.datetime.now()))
            self.SOM.fit(X=X, **SOM_params)

        # Cluster training
        if Cluster_params is not None:
            if verboses == 1: print("Start of clustering SOM prototypes at {}".format(datetime.datetime.now()))

            # Search for k_opt
            k_opt = None
            if type(Cluster_params['n_clusters']) is dict:  # a search is implied:
                eval_function = Cluster_params['n_clusters']['metric']
                find_best = Cluster_params['n_clusters']['criteria']
                k_values = Cluster_params['n_clusters']['k_values']

                validation_index = [0] * len(k_values)
                for i in range(len(k_values)):
                    kmeans = KMeans(n_clusters=k_values[i],
                                    n_init=10,
                                    init='random'
                                    # n_jobs=-1
                                    ).fit(self.SOM.neurons)
                    # test if number of distinct clusters == number of clusters specified
                    centroids = kmeans.cluster_centers_
                    if len(centroids) == len(np.unique(centroids, axis=0)):
                        validation_index[i] = eval_function(kmeans, self.SOM.neurons)
                    else:
                        validation_index[i] = np.NaN

                k_opt = k_values[find_best(validation_index)]
                if verboses == 1: print("Best k found: {}".format(k_opt))
            else:
                k_opt = Cluster_params['n_clusters']

            params = Cluster_params.copy()
            del params['n_clusters']  # deleting unecessary param
            # real training of clustering algorithm
            self.Cluster = KMeans(n_clusters=k_opt, **params).fit(self.SOM.neurons)

        # Model training
        self.region_labels = self.regionalize(X)  # finding labels of datapoints
        if verboses == 1: print("Start of Model training at {}".format(datetime.datetime.now()))

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
        # searching for a non-empty region
        regions = [i for i in range(self.Cluster.n_clusters)]
        not_empty_regions = list(set(regions) - set(self.empty_regions))

        #         temp = self.regional_models[not_empty_regions[0]].intercept_
        predictions = np.zeros((
            len(X),  # number of samples
            self.targets_dim_  # size of output Y
        ))

        regions = self.regionalize(X)
        for i in range(len(X)):
            predictions[i, :] = self.regional_models[regions[i]].predict(X[i].reshape(1, -1))

        return predictions
