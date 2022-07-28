# TODO: __author__ = "Joaquim Viegas"

""" JQM_CV - Python implementations of Dunn and Davis Bouldin clustering validity indices

dunn(k_list):
    Slow implementation of Dunn index that depends on numpy
    -- basec.pyx Cython implementation is much faster but flower than dunn_fast()
dunn_fast(points, labels):
    Fast implementation of Dunn index that depends on numpy and sklearn.pairwise
    -- No Cython implementation
davisbouldin(k_list, k_centers):
    Implementation of Davis Boulding index that depends on numpy
    -- basec.pyx Cython implementation is much faster
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000

    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])

    return np.min(values)

def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])

    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])

    return np.max(values)


def dunn(k_list):
    """ Dunn index [CVI]
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))

    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])

        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di


def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)


def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]

    return np.max(values)


def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))

    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])

    l_range = list(range(0, len(ks)))

    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)

        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di


def  big_s(x, center):
    len_x = len(x)
    total = 0

    for i in range(len_x):
        total += np.linalg.norm(x[i]-center)

    return total/len_x


def davisbouldin(k_list, k_centers):
    """ Davis Bouldin Index
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    k_centers : np.array
        The array of the cluster centers (prototypes) of type np.array([K, p])
    """
    len_k_list = len(k_list)
    big_ss = np.zeros([len_k_list], dtype=np.float64)
    d_eucs = np.zeros([len_k_list, len_k_list], dtype=np.float64)
    db = 0

    for k in range(len_k_list):
        big_ss[k] = big_s(k_list[k], k_centers[k])

    for k in range(len_k_list):
        for l in range(0, len_k_list):
            d_eucs[k, l] = np.linalg.norm(k_centers[k]-k_centers[l])

    for k in range(len_k_list):
        values = np.zeros([len_k_list-1], dtype=np.float64)
        for l in range(0, k):
            values[l] = (big_ss[k] + big_ss[l])/d_eucs[k, l]
        for l in range(k+1, len_k_list):
            values[l-1] = (big_ss[k] + big_ss[l])/d_eucs[k, l]

        db += np.max(values)
    res = db/len_k_list
    return res


# --------------------------------------- Implemented by Rômulo ------------------------------------------------------ #
from numpy.linalg import norm


def DB(model, X, q=2, t=2):
    k = len(model.cluster_centers_)  # number of clusters
    db = 0
    for i in range(k):
        db += R(i, q, t, model, X)
    db /= k
    return db


def R(i, q, t, model, X):
    js = [j for j in range(len(model.cluster_centers_)) if j != i]  # for j!=i
    R_iqt = 0  # R_iqt is always >= 0
    for j in js:
        temp = (S(i, q, model, X) + S(j, q, model, X)) / d(i, j, t, model)
        R_iqt = temp if temp > R_iqt else R_iqt  # searching for the maximum
    return R_iqt


def S(i, q, model, X):
    Vi = X[np.where(model.labels_ == i)]  # partition 'i'
    wi = model.cluster_centers_[i]  # cluster center of partition 'i'
    S_iq = 0
    for x in Vi:
        S_iq += norm(x - wi) ** q

    S_iq = (S_iq / len(Vi)) ** (1 / q)
    return S_iq


def d(i, j, t, model):
    wi = model.cluster_centers_[i]
    wj = model.cluster_centers_[j]
    d_ijt = 0
    for m in range(len(wi)):
        d_ijt += (abs(wi[m] - wj[m])) ** t
    d_ijt = d_ijt ** (1 / t)
    return d_ijt


# %%

from numpy import trace


def CH(kmeans, points):
    # ks is the vector of clusters. Ex: [0 1 2 3 4]
    # N is the vector frequency of each label. Ex: [14 10 5 12]
    ks, N = np.unique(np.sort(kmeans.labels_), return_counts=True)
    x_bar = np.mean(points, axis=0).reshape(-1, 1)

    # matriz dispersão entregrupos
    Bk = np.zeros((points.shape[1], points.shape[1]))
    # matriz dispersão intragrupo
    Wk = np.zeros((points.shape[1], points.shape[1]))
    for i in ks:  # para cada protótipo
        wi = kmeans.cluster_centers_[i].reshape(-1, 1)  # protótipo do cluster 'i'
        temp = wi - x_bar
        Bk += N[i] * np.matmul(temp, temp.T)

        Vi = points[kmeans.labels_ == i].T
        for l in range(Vi.shape[1]):  # para cada elemento da partição 'i'
            temp = Vi[:, l] - wi
            Wk += np.matmul(temp, temp.T)

    N = len(points)  # number of points
    K = len(kmeans.cluster_centers_)  # number os clusters
    ch = (trace(Bk) / (K - 1)) / (trace(Wk) / (N - K))
    return ch
