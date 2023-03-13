from collections import OrderedDict
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from torch import load,cat
from sklearn import manifold, datasets

# Dataset -----------------------------------------------------------------------
PATH = '/home/osvaldo/Documents/CCNY/Project_RBP/results/int_outputs/'
eval_exp = '22'
data = []
color = []

for batch_idx in range(100):
    name_1 = 'ev_' + eval_exp + '/ev_' + eval_exp + '_batch_' + "{:02d}".format(batch_idx) + '.pt'
    name_2 = 'ev_' + eval_exp + '/ev_' + eval_exp + '_target_' + "{:02d}".format(batch_idx) + '.pt'

    data.append(load(PATH+name_1))
    color.append(load(PATH+name_2))

data = cat(data).to('cpu')
color = cat(color).to('cpu')

# Set-up manifold methods -------------------------------------------------------
n_components = 2
method = manifold.TSNE(n_components=n_components, init="pca", random_state=0)
Y = method.fit_transform(data)

# Create figure -----------------------------------------------------------------
fig,axs  = plt.subplots(figsize=(10,10))

# Plot results
im = axs.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
fig.colorbar(im, ax=axs)
axs.xaxis.set_major_formatter(NullFormatter())
axs.yaxis.set_major_formatter(NullFormatter())
axs.axis("tight")

figname = '/home/osvaldo/Documents/CCNY/Project_RBP/results/plots/t_sne_' + eval_exp + '.svg'
fig.savefig(figname,format='svg')

#plt.show()
