import datetime
import numpy as np

from copy import copy


class BiasModel:
    """Class the implements a dummy model in case of homogenous region"""

    def __init__(self, class_label):
        self.class_label = class_label

    def predict(self, X):
        return np.vstack([self.class_label] * len(X))


class LocalModel:
    """Class of Local Models."""

    def __init__(self, ClusterAlg, ModelAlg):
        self.ClusterAlg = ClusterAlg
        self.ModelAlg   = ModelAlg

        self.clusters = None
        self.models = None
        self.targets_dim_ = None

        self.empty_regions = []

    def fit(self, X, Y, Cluster_params=None, verboses=0):
        self.targets_dim_ = Y.shape[1]  # dimension of target values

        # Clustering
        if verboses == 1:
            print("Start of clusterization: {}".format(datetime.datetime.now()))

        if Cluster_params:
            self.ClusterAlg = self.ClusterAlg(**Cluster_params)

        self.ClusterAlg.fit(X)
        self.clusters = self.ClusterAlg.cluster_centers_

        # Local models training
        if verboses == 1:
            print("Start of local models training: {}".format(datetime.datetime.now()))

        n_clusters  = self.ClusterAlg.n_clusters
        labels      = self.ClusterAlg.labels_  # labels of each datapoints
        self.models = [{}] * n_clusters

        for i in range(n_clusters):  # for each region
            Xi = X[np.where(labels == i)[0]]
            Yi = Y[np.where(labels == i)[0]]

            model_i = None
            local_classes = np.unique(Yi, axis=0)  # list of local classes
            if len(Xi) == 0:  # empty region
                self.empty_regions.append(i)

            elif len(local_classes) == 1:  # homogenous region, only one class
                model_i = BiasModel(local_classes)

            else:  # region not empty or homogeneous
                self.ModelAlg.fit(Xi, Yi)
                model_i = copy(self.ModelAlg)

            self.models[i] = model_i

    def predict(self, X, rounded=False):
        predictions = np.zeros((
            len(X),  # number of samples
            self.targets_dim_  # size of output Y
        ))

        region_list = self.ClusterAlg.predict(X)  # .astype(int) # list of regions of each sample
        for i in range(len(X)):
            predictions[i, :] = self.models[region_list[i]].predict(X[i].reshape(1, -1))
            # should I worry about regions with no models? (empty regions in training)
            # isn't that a sign of bad clustering?

        return predictions if not rounded else np.round(np.clip(predictions, 0, 1))

    def find_non_empty_regions(self, X):
        # TODO: Check if it is useful
        region_list = self.ClusterAlg.predict(X)  # .astype(int) # list of regions of each sample

        for i in range(len(X)):  # for each datapoint
            # if the region don't have a model is because it didn't have datapoints in the train set
            if region_list[i] in self.empty_regions:
                # calculate distance from all clusters
                dist_2 = np.sum((self.clusters - X[i]) ** 2, axis=1)  # euclidian_norm**2
                dist_2[self.empty_regions] = np.inf  # taking off dead clusters from the play

                temp = np.argmin(dist_2)
                region_list[i] = temp

        return region_list