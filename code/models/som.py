import numpy as np
import plotly.offline as plt
import plotly.graph_objs as go
import ipywidgets as widgets

from math import ceil
from IPython.display import clear_output

plt.init_notebook_mode(connected=True)      # Enabling plotly inside jupyter notebook


class SOM:
    """ Class of Self Organizing Maps conected in a two-dimensional grid. """

    def __init__(self, nRows, nColumns):
        self.nRows = nRows
        self.nColumns = nColumns
        self.__epochs = 0  # number of epochs of trained SOM

        self.neurons = None
        self.indexes = None  # list of index2D <=> index1D

        self.neuronsHist = None
        self.ssdHist = None

    def init(self, X):  # giving the data, so we can define maximum and minimum in each dimension
        # reset neuronsHist and ssdHist
        self.neuronsHist = None
        self.ssdHist = None

        dim = X.shape[1]  # number of features
        rand = np.random.rand(self.nRows * self.nColumns, dim)  # Auxiliary random element
        # find min-max for each dimension
        minimum = np.amin(X, axis=0)
        maximum = np.amax(X, axis=0)
        # Initializing neurons in random positions between minimum and maximum of the dataset
        self.neurons = (maximum - minimum) * rand + minimum

        # list of index2D == index1D
        self.indexes = self.index1Dto2D(np.arange(len(self.neurons)))

    def fit(self, X, alpha0, sigma0, nEpochs=100, saveNeuronsHist=False, saveSSDHist=True, tol=1e-6,
            verboses=0, batchSize=np.inf):
        if self.neurons is None: self.init(X)  # if self.init() wasn't run

        tau1 = nEpochs / sigma0
        tau2 = nEpochs
        SSD_new = self.SSD(X)  # initial SSD, from random parameters

        if saveNeuronsHist:
            self.neuronsHist = [np.zeros(self.neurons.shape)] * (nEpochs + 1)
            self.neuronsHist[0] = np.copy(self.neurons)  # random neurons
        if saveSSDHist:
            self.ssdHist = np.zeros((nEpochs + 1))
            self.ssdHist[0] = SSD_new  # initial SSD, from random parameters

        sigma   = sigma0
        alpha   = alpha0
        inertia = np.inf  # initial value of inertia

        batchSize = min(len(X), batchSize)  # adjusting ill defined batchSize
        for epoch in range(nEpochs):
            # Updating alpha and sigma
            sigma = sigma0 * np.exp(-epoch / tau1)
            alpha = alpha0 * np.exp(-epoch / tau2)

            order = np.random.permutation(len(X))  # shuffling order
            for i in order[:batchSize]:  # for each datapoint in the shuflled order until batchSize
                # search for winner neuron
                winner_idx = self.get_winner(X[i])
                # neighbor function
                h_ik = self.h_neighbor(winner_idx, self.indexes, sigma)
                # updating neurons
                self.neurons += (h_ik[:, np.newaxis] * (X[i] - self.neurons)) * alpha

            self.__epochs += 1  # updating number of epochs trained

            if verboses == 1:
                print("End of epoch {}".format(epoch + 1))

            SSD_old = SSD_new
            SSD_new = self.SSD(X)
            inertia = abs((SSD_old - SSD_new) / SSD_old)

            # Saving if necessary
            if saveNeuronsHist:
                self.neuronsHist[epoch + 1] = np.copy(self.neurons)

            if saveSSDHist:
                self.ssdHist[epoch + 1] = SSD_new

            # breaking if tolerance reached before nEpochs
            if inertia < tol:
                # history cutting
                if saveNeuronsHist:
                    self.neuronsHist = self.neuronsHist[0:epoch + 2]
                if saveSSDHist:
                    self.ssdHist = self.ssdHist[0:epoch + 2]
                break

    def SSD(self, X):
        SSD = 0
        for x in X:  # can it be vectorized?
            dist_2 = np.sum((self.neurons - x) ** 2, axis=1)  # norm**2
            SSD += np.amin(dist_2)
        return SSD

    def index1Dto2D(self, index):  # convert index of neurons parameter matrix to the 2D grid index
        return np.asarray([np.ceil((index + 1) / self.nRows) - 1, index % self.nRows], dtype='int').T

    def get_winner(self, x, dim=2):
        dist_2 = np.sum((self.neurons - x) ** 2, axis=1)  # norm**2
        temp = np.argmin(dist_2)
        winner = self.index1Dto2D(temp) if dim == 2 else temp
        return winner

    def h_neighbor(self, idx_1, idx_2, sigma):
        dist_2 = np.sum((idx_2 - idx_1) ** 2, axis=1)  # norm**2
        return np.exp(-dist_2 / (2 * sigma ** 2))

    def getLabels(self, X, dim=1):  # get labels in 1 dimension or 2 dimension (neuron grid)
        N = len(X)
        labels = np.zeros((N, dim), dtype='int')
        for i in range(N):
            labels[i, :] = self.get_winner(X[i, :], dim)
        return labels

    def plotSSD(self):
        traceData = go.Scatter(
            x=[i + 1 for i in range(self.__epochs)],  # epochs
            y=self.ssdHist,
            mode='lines',
            name='SSD')
        data = [traceData]
        layoutData = go.Layout(
            title="SSD history",
            xaxis=dict(title='Epoch'),
            yaxis=dict(title='SSD')
        )

        fig = go.Figure(data=data, layout=layoutData)
        plt.iplot(fig)

    def plotSOM(self, X=None):
        fig = go.Figure()

        n_epochs = self.__epochs

        traces = []
        shapes = []

        # Function to draw the graph
        for nth_it in range(0, n_epochs):
            x = self.neuronsHist[nth_it][:, 0].tolist()
            y = self.neuronsHist[nth_it][:, 1].tolist()
            name = 'neurons [epoch =' + str(nth_it) + ']'

            neurons = go.Scatter(x=x, y=y, mode='markers', name=name, marker=dict(size=10, color='#673AB7'))

            traces.append(neurons)

            # Arestas que conectam os neurônios
            linhas = [{}] * (2 * self.nRows * self.nColumns - self.nRows - self.nColumns)

            # Contador de linhas
            count = 0

            for linha in range(self.nRows):  # conecta da esquerda para direita
                for coluna in range(self.nColumns):  # e de cima para baixo
                    try:
                        if self.neuronsHist is not None:
                            x0 = self.neuronsHist[nth_it][np.where(
                                (self.indexes == (linha, coluna)).all(axis=1))[0][0], 0]
                            y0 = self.neuronsHist[nth_it][np.where(
                                (self.indexes == (linha, coluna)).all(axis=1))[0][0], 1]
                            x1 = self.neuronsHist[nth_it][np.where(
                                (self.indexes == (linha, coluna + 1)).all(axis=1))[0][0], 0]
                            y1 = self.neuronsHist[nth_it][np.where(
                                (self.indexes == (linha, coluna + 1)).all(axis=1))[0][0], 1]
                        else:
                            x0 = self.neurons[np.where((self.indexes == (linha, coluna)).all(axis=1))[0][0], 0]
                            y0 = self.neurons[np.where((self.indexes == (linha, coluna)).all(axis=1))[0][0], 1]
                            x1 = self.neurons[np.where((self.indexes == (linha, coluna + 1)).all(axis=1))[0][0], 0]
                            y1 = self.neurons[np.where((self.indexes == (linha, coluna + 1)).all(axis=1))[0][0], 1]

                        linhas[count] = {'type': 'line', 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                                         'line': {'color': '#673AB7', 'width': 1, }}
                        count += 1
                    except:  # edge of the grid
                        pass
                    try:
                        if self.neuronsHist is not None:
                            x0 = self.neuronsHist[nth_it][np.where(
                                (self.indexes == (linha, coluna)).all(axis=1))[0][0], 0]
                            y0 = self.neuronsHist[nth_it][np.where(
                                (self.indexes == (linha, coluna)).all(axis=1))[0][0], 1]
                            x1 = self.neuronsHist[nth_it][np.where(
                                (self.indexes == (linha + 1, coluna)).all(axis=1))[0][0], 0]
                            y1 = self.neuronsHist[nth_it][np.where(
                                (self.indexes == (linha + 1, coluna)).all(axis=1))[0][0], 1]
                        else:
                            x0 = self.neurons[np.where((self.indexes == (linha, coluna)).all(axis=1))[0][0], 0]
                            y0 = self.neurons[np.where((self.indexes == (linha, coluna)).all(axis=1))[0][0], 1]
                            x1 = self.neurons[np.where((self.indexes == (linha + 1, coluna)).all(axis=1))[0][0], 0]
                            y1 = self.neurons[np.where((self.indexes == (linha + 1, coluna)).all(axis=1))[0][0], 1]

                        linhas[count] = {'type': 'line', 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                                         'line': {'color': '#673AB7', 'width': 1}}
                        count += 1
                    except:  # edge of the grid
                        pass

            shapes.append(linhas)

            # layout    = go.Layout(title=title, xaxis=dict(title="$x_1$"), yaxis=dict(title="$x_2$"), shapes=linhas)
            # fig_ith   = go.Figure(data=data_ith, layout=layout)
            #
            # fig.data[nth_it] = fig_ith.data

        datapoints = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', name='data',
                                marker=dict(size=5, color='#03A9F4'))

        layout = go.Layout(title="title", xaxis=dict(title="$x_1$"), yaxis=dict(title="$x_2$"), shapes=shapes[0])
        data = go.Data([datapoints] + traces)
        fig = go.Figure(data=data, layout=layout)

        for dt in fig.data:
            dt.visible = False

        fig.data[0].visible = True
        fig.data[1].visible = True

        # shapes_all = [shapes[0]] + shapes

        steps = []
        for i in range(0, n_epochs):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": "Slider switched to step: " + str(i), "shapes": shapes[i]}],  # layout attribute
            )

            step["args"][0]["visible"][0] = True  # Toggle i'th trace to "visible"

            if i != 0:
                step["args"][0]["visible"][i + 1] = True  # Toggle i'th trace to "visible"
            else:
                step["args"][0]["visible"][1] = True
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Epoch: "},
            pad={"t": n_epochs},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders
        )

        fig.show()
