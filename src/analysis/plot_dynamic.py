# Plot dynamic

import torch
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np 

PATH_data = '/media/osvaldo/OMV5TB/temporal_outputs/'

batch_idx = 10
data = {'exp_03':{'wo':[],'w':[]},'exp_14':{'wo':[],'w':[]}}
target = {'exp_03':{'wo':[],'w':[]},'exp_14':{'wo':[],'w':[]}}

pooling = torch.nn.MaxPool2d(16)
n_components = 2
method = manifold.TSNE(n_components=n_components, init="pca", random_state=0)

# ------------------------------------------------------------------------
for exptitle in ['exp_03','exp_14']:
	num_steps = 20 if exptitle=='exp_14' else 1
	for cc in ['wo','w']:
		filename = PATH_data + exptitle + '_' + cc + '_noise/' + exptitle + '_target_' + "{:02d}".format(batch_idx) + '.pt'
		target[exptitle][cc] = torch.load(filename)

		for tt in range(num_steps):
			filename = PATH_data + exptitle + '_' + cc + '_noise/' + exptitle + '_batch_' + "{:02d}".format(batch_idx) + '_t_' + "{:02d}".format(tt) + '.pt'
			X = torch.load(filename)
			data[exptitle][cc].append(pooling(X).view(-1,128))

		data[exptitle][cc] = torch.cat(data[exptitle][cc]).to('cpu')
# ------------------------------------------------------------------------
X = [ww2 for kk,ww in data.items() for kk2,ww2 in ww.items()]
order = [kk + '_' + kk2 for kk,ww in data.items() for kk2,ww2 in ww.items()]

X = torch.cat(X)
Y = method.fit_transform(X)

Ydata = {'exp_03':{'wo':Y[:100,:],'w':Y[100:200,:]},'exp_14':{'wo':Y[200:2200,:],'w':Y[2200:4200,:]}}

# ------------------------------------------------------------------------

fig, axs = plt.subplots(2,2)

# EXP 03 -----------------------------------------------------------------
exptitle = 'exp_03'
j=0
for cc in ['wo','w']:
	R = Ydata[exptitle][cc]
	axs[0,j].scatter(R[:,0],R[:,1],c=target[exptitle][cc].to('cpu'), cmap='plasma')
	j+=1

# EXP 14 -----------------------------------------------------------------
colors = plt.cm.plasma(np.linspace(0, 1, 10))

exptitle = 'exp_14'
num_steps = 20
j=0
for cc in ['wo','w']:
	R = Ydata[exptitle][cc]

	for ii in range(100):
		index = [100*jj + ii for jj in range(num_steps)]
		axs[1,j].plot(R[index,0],R[index,1], c=colors[target[exptitle][cc].to('cpu')[ii]])

	j+=1

plt.show()