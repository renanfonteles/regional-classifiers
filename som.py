import numpy as np
import plotly.offline as plt
import plotly.graph_objs as go
from math import ceil
from random import randint
import ipywidgets as widgets
from IPython.display import clear_output
from plotly import tools
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler, StandardScaler

plt.init_notebook_mode(connected=True) # enabling plotly inside jupyter notebook


def scale_feat(X_train, X_test, scaleType='min-max'):
    if scaleType=='min-max' or scaleType=='std':
        X_tr_norm = np.copy(X_train) # fazendo cópia para deixar original disponível
        X_ts_norm = np.copy(X_test)
        scaler = MinMaxScaler() if scaleType=='min-max' else StandardScaler()
        scaler.fit(X_tr_norm)
        X_tr_norm = scaler.transform(X_tr_norm)
        X_ts_norm = scaler.transform(X_ts_norm)
        return (X_tr_norm, X_ts_norm)
    else:
        raise ValueError("Tipo de escala não definida. Use 'min-max' ou 'std'.")


class SOM_2D:
    'Class of Self Organizing Maps conected in a two-dimensional grid.'
    
    def __init__(self, nRows, nColumns, dim): 
        self.nRows    = nRows
        self.nColumns = nColumns
        self.dim = dim   # neurons dimension = features dimension
        self.nEpochs = 0 # number of epochs of trained SOM
        
        self.param = np.zeros((dim, nRows, nColumns))
        self.paramHist = None
        self.ssdHist   = None
        
    def init(self, X): # giving the data, so we can define maximum and minimum in each dimension
        self.paramHist = None # reset paramHist and ssdHist
        self.ssdHist   = None
        
        # Auxiliary random element
        rand_01 = np.random.rand(self.dim, self.nRows, self.nColumns)
        # find min-max for each dimension:
        minimum = np.amin(X, axis=0)
        maximum = np.amax(X, axis=0)
        for dim in range(self.dim):
            self.param[dim,:,:] = (maximum[dim]-minimum[dim])*rand_01[dim,:,:] + minimum[dim]
        
    
    #def update_neuron(self, args):
    #    row, column, winner_idx, alpha, sigma, i = args
    #    h_ik = self.h_neighbor(winner_idx, [row, column], sigma)
    #    self.param[:,row,column] += alpha * h_ik * (X[i] - self.param[:,row,column])
        
    
    def train(self, X, alpha0, sigma0, nEpochs=100, batchSize=100, saveParam=False, saveSSD=True, tol=1e-6,
              verboses=0):
        tau1 = nEpochs/sigma0
        tau2 = nEpochs
        SSD_new = self.SSD(X) # initial SSD, from random parameters
        
        if saveParam: 
            self.paramHist = np.zeros((nEpochs+1, self.dim, self.nRows, self.nColumns))
            self.paramHist[0,:,:,:] = self.param # random parameters
        if saveSSD:
            self.ssdHist = np.zeros((nEpochs+1))
            self.ssdHist[0] = SSD_new # initial SSD, from random parameters
        
        sigma = sigma0
        alpha = alpha0
        inertia = np.inf # initial value of inertia
        batchSize = X.shape[0] if X.shape[0] < batchSize else batchSize # adjusting ill defined batchSize
        for epoch in range(nEpochs):
            # Updating alpha and sigma
            sigma = sigma0*np.exp(-epoch/tau1);
            alpha = alpha0*np.exp(-epoch/tau2);
            
            # shuffled order
            order = np.random.permutation(X.shape[0])
            for i in range(batchSize):
                # search for winner neuron
                winner_idx = self.get_winner(X[i])
                
                # updating neurons weights
                #args = [(r,c,winner_idx,alpha,sigma,i) for r in range(self.nRows) for c in range(self.nColumns)]
                
                #pool = Pool()                      # Create a multiprocessing Pool
                #pool.map(self.update_neuron, args) # process data_inputs iterable with pool
                #pool.close()
                #pool.join()
                #print("end of tata point: {}".format(i))
                
                
                for row in range(self.nRows):
                    for column in range(self.nColumns):
                        h_ik = self.h_neighbor(winner_idx, [row, column], sigma)
                        self.param[:,row,column] += alpha * h_ik * (X[i] - self.param[:,row,column])
                
               
            
            
            self.nEpochs = epoch+1 # saving number of epochs
            if verboses==1:
                print("End of epoch {}".format(self.nEpochs))
            
            SSD_old = SSD_new
            SSD_new = self.SSD(X)
            inertia = abs((SSD_old - SSD_new)/SSD_old)
            
            # Saving if necessary
            if saveParam:
                self.paramHist[epoch+1,:,:,:] = self.param
            if saveSSD:
                self.ssdHist[epoch+1] = SSD_new
                       
            if inertia < tol: # maybe break before nEpochs
                # history cutting
                if saveParam:
                    self.paramHist = self.paramHist[0:epoch+2,:,:,:]
                if saveSSD:
                    self.ssdHist = self.ssdHist[0:epoch+2]
                
                break
            
            
    def SSD(self, X):
        SSD = 0
        for x in X:
            dist_min = np.inf
            for row in range(self.nRows):
                for column in range(self.nColumns):
                    temp = x - self.param[:,row,column]
                    dist = np.dot(temp,temp)
                    if dist < dist_min:
                        dist_min = dist
            SSD += dist_min
        return SSD
        
        
    def get_winner(self, x):
        dist_matrix = np.zeros((self.nRows, self.nColumns)) # norm**2
        for row in range(self.nRows):
            for column in range(self.nColumns):
                aux = x - self.param[:,row,column]
                dist_matrix[row,column] = np.dot(aux,aux)
        result = [ceil((dist_matrix.argmin()+1)/self.nRows)-1, dist_matrix.argmin()%self.nRows]        
        return result
    
    
    def h_neighbor(self, idx_1, idx_2, sigma):
        aux = np.asarray(idx_1) - np.asarray(idx_2)
        return np.exp( -np.dot(aux,aux)/(2*sigma**2) )
    
    def getLabels(self, X):
        N = len(X)
        labels = np.zeros((N,2))
        #labels = [self.get_winner(X[i,:]) for i in range(len(X))]
        for i in range(N):
            labels[i,:] = self.get_winner(X[i,:])
            
        return labels
    
    def plotSSD(self):
        traceData = go.Scatter(
            x = [i+1 for i in range(self.nEpochs)], # epochs
            y = self.ssdHist, 
            mode='lines',
            name='SSD')
        data = [traceData]
        layoutData = go.Layout(
            title = "SSD history",
            xaxis=dict(title='Epoch'),
            yaxis=dict(title='SSD')
        )

        fig = go.Figure(data=data, layout=layoutData)
        plt.iplot(fig)

    def paramAsMatrix(self): # return the 3D matrix of param as a 2D matrix as in k-means
        som_clusters = np.zeros((self.nRows*self.nColumns, self.dim))
        count=0
        for r in range(self.nRows):
            for c in range(self.nColumns):
                som_clusters[count] = self.param[:,r,c]
                count+=1
        return som_clusters
    
    
   
 
# function to plot SOM in the special case when the feature space is 2D
def plot_SOM(SOM, X): 
    if SOM.paramHist is not None:
        # Int box to change the iteration number
        n_txt = widgets.BoundedIntText(
            value=0,
            min=0,
            max=len(SOM.paramHist)-1,
            step=10,
            description='epoch:'
        )    
        
    # Function to draw the graph
    def atualizarGrafico(change):
        clear_output()

        if SOM.paramHist is not None:
            display(n_txt)    
            n_ = change['new'] # new iteration number
    
        if X is not None:
            datapoints = go.Scatter(
                x = X[:,0], 
                y = X[:,1], 
                mode='markers',
                name='data',
                marker = dict(
                     size = 5,
                     color = '#03A9F4'
                    )
            )

            if SOM.paramHist is not None:
                x = SOM.paramHist[n_,0,:,:].reshape(-1).tolist() 
                y = SOM.paramHist[n_,1,:,:].reshape(-1).tolist()
                name = 'neurons [epoch ='+str(n_)+']'
            else:
                x = SOM.param[0,:,:].reshape(-1).tolist()
                y = SOM.param[1,:,:].reshape(-1).tolist()
                name = 'neurons'
            
            neurons = go.Scatter(x=x, y=y, mode='markers', name=name, 
                                 marker = dict(size=10,color = '#673AB7'))

            data = [datapoints, neurons]

            # cada linha que conecta os neurônios
            linhas = [{}]*(2*SOM.nRows*SOM.nColumns - SOM.nRows - SOM.nColumns)
            count=0 #contador para saber qual linha estamos
            for linha in range(SOM.nRows): # conecta da esquerda para direita
                for coluna in range(SOM.nColumns): # e de cima para baixo
                    try:
                        if SOM.paramHist is not None:
                            x0 = SOM.paramHist[n_,0,linha,coluna]
                            y0 = SOM.paramHist[n_,1,linha,coluna]
                            x1 = SOM.paramHist[n_,0,linha,coluna+1]
                            y1 = SOM.paramHist[n_,1,linha,coluna+1]
                        else:
                            x0 = SOM.param[0,linha,coluna]
                            y0 = SOM.param[1,linha,coluna]
                            x1 = SOM.param[0,linha,coluna+1]
                            y1 = SOM.param[1,linha,coluna+1]
                            
                        linhas[count]= {'type':'line','x0':x0,'y0': y0,'x1':x1,'y1':y1,
                                        'line': {'color': '#673AB7','width': 1,}}
                        count+=1
                    except:
                        pass
                    try:
                        if SOM.paramHist is not None:
                            x0 = SOM.paramHist[n_,0,linha,coluna]
                            y0 = SOM.paramHist[n_,1,linha,coluna]
                            x1 = SOM.paramHist[n_,0,linha+1,coluna]
                            y1 = SOM.paramHist[n_,1,linha+1,coluna]
                        else:
                            x0 = SOM.param[0,linha,coluna]
                            y0 = SOM.param[1,linha,coluna]
                            x1 = SOM.param[0,linha+1,coluna]
                            y1 = SOM.param[1,linha+1,coluna]
                        
                        linhas[count] = {'type': 'line','x0': x0,'y0': y0,'x1': x1,'y1': y1,
                                         'line': {'color': '#673AB7','width': 1}}
                        count+=1
                    except:
                        pass

            layout = go.Layout(
                title = "Dados + SOM",
                xaxis=dict(title="$x_1$"),
                yaxis=dict(title="$x_2$"),
                shapes=linhas
            )

            fig = go.Figure(data=data, layout=layout)
            plt.iplot(fig)
    
    if SOM.paramHist is not None:
        n_txt.observe(atualizarGrafico, names='value')
        
    atualizarGrafico({'new': 0})