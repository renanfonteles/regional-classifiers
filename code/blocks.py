from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def data_preprocessing(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    temp   = scaler.fit_transform(data.data)

    pca = PCA(n_components=2)
    X   = pca.fit_transform(temp)

    return X
