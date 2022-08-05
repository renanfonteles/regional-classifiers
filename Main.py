from code.models.som import SOM
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go
import plotly.offline as pyo

pyo.init_notebook_mode(connected=True)      # Enabling plotly inside jupyter notebook

data = load_digits()

scaler = MinMaxScaler()
temp   = scaler.fit_transform(data.data)

pca = PCA(n_components=2)
X = pca.fit_transform(temp)

fig = go.Figure(data=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers'))
fig.update_layout({
    'title': 'MNIST data set after PCA (2 components)'
})

fig.show("notebook")

N = len(X)
l = int( ( 5*(N)**(1/2) )**(1/2) ) # side of square grid
som = SOM(l, l)
som.fit(X, alpha0=0.2, sigma0=4, nEpochs=30, saveNeuronsHist=True)

som.plotSOM(X)