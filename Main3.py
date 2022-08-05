import numpy as np
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.figure_factory as ff

z = [[.1, .3, .5, .7],
     [1, .8, .6, .4],
     [.6, .4, .2, .0],
     [.9, .7, .5, .3]]

Index1 = ['ARI', 'AMI', 'VM', 'FM', 'SI', 'CH', 'DB', 'DU', 'FPE', 'AIC', 'BIC', 'MDL']
Cols  = ['pk', 'vc2c', 'vc3c', 'wf2f', 'wf4f', 'wf24f']

data = [[16, 21, 21, 13, 12, 14, 10, 13, 12, 12, 12, 12], [16, 14, 7, 19, 19, 19, 12, 8, 11, 8, 13, 6], [20, 8, 15, 20, 20, 20, 15, 7, 7, 6, 10, 9], [2, 2, 2, 2, 0, 21, 2, 2, 29, 29, 28, 28], [6, 4, 8, 11, 11, 10, 7, 8, 10, 10, 9, 10], [26, 0, 0, 16, 0, 16, 0, 0, 1, 1, 5, 2]]
data_np = np.array(data)

df1 = DataFrame(data_np, index=Cols, columns=Index1)

ax1 = sns.heatmap(df1, annot=True, cmap="coolwarm", cbar_kws={'label': 'Hit-rate of $K_{opt}$'})
ax1.set_ylabel('Datasets')
ax1.set_xlabel('Indices acronyms')

plt.savefig('fig1_local.pdf')
plt.show()

data_regional = [[21, 23, 23, 28, 21, 21, 21, 21], [24, 25, 15, 16, 20, 18, 28, 15], [24, 22, 20, 10, 23, 22, 28, 21], [0, 28, 4, 14, 33, 33, 31, 33], [7, 8, 7, 12, 11, 12, 15, 11], [21, 24, 0, 7, 24, 24, 24, 24]]
data_np_reg   = np.array(data_regional)

Index2 = ['SI', 'CH', 'DB', 'DU', 'FPE', 'AIC', 'BIC', 'MDL']

df2 = DataFrame(data_np_reg, index=Cols, columns=Index2)

ax2 = sns.heatmap(df2, annot=True, cmap="viridis", cbar_kws={'label': 'Hit-rate of $K_{opt}$'})
ax2.set_ylabel('Datasets')
ax2.set_xlabel('Indices acronyms')

plt.savefig('fig2_regional.pdf')
plt.show()