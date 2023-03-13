import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------

PATH = '/home/osvaldo/Documents/CCNY/Project_RBP/results_detection/training-evolution/'

exp_metrics = {'exp_34':None,
			'exp_35':None,
			'exp_36':None,
			'exp_37':None,}

loss = {'exp_34':None,
		'exp_35':None,
		'exp_36':None,
		'exp_37':None}			

num_metrics_per_epoch = 12
# -------------------------------------------------------------------

colors = np.concatenate((plt.cm.Blues(np.linspace(0,1,num_metrics_per_epoch//2)),
						plt.cm.Reds(np.linspace(0,1,num_metrics_per_epoch//2))),
						axis=0)

faster_rcnn_metrics = [0.369,0.585,0.396,0.212,0.403,0.482,0.307,0.485,0.509,0.317,0.544,0.649]

for name in exp_metrics.keys():
	f = open(PATH + name + '_terminal.txt', "r")
	data = [ll for ll in f.readlines() if ll[0]==' ']
	exp_metrics[name] = np.array([float(ll[-6:-1]) for ll in data]).reshape((-1,num_metrics_per_epoch))
	loss[name] = np.loadtxt(PATH + name + '.dat')

metrics_name = [ll[:-9] for ll in data[0:num_metrics_per_epoch]]
# -------------------------------------------------------------------

metrics_plot = [3,4,5,9,10,11]
metrics_table = [0,1,2,6,7,8]

# -------------------------------------------------------------------
fig1, axs1 = plt.subplots(3,2,figsize=(10,15))
fig2, axs2 = plt.subplots(1,1,figsize=(5,5))
ylim_colums = [0.4,0.6]

# -------------------------------------------------------------------
for nn,mm in enumerate(metrics_plot):
	ii = nn%3
	jj = nn//3
	for ee, kk in enumerate(exp_metrics):
		axs1[ii,jj].plot(1+np.arange(26),exp_metrics[kk][:,mm], label=kk)

	axs1[ii,jj].set_ylim(0,ylim_colums[jj])
	axs1[ii,jj].legend()
	axs1[ii,jj].set_title(metrics_name[mm])

# -------------------------------------------------------------------
for nn, kk in enumerate(loss):
	axs2.plot(1+loss[kk][0,:],loss[kk][1,:],label = kk)

axs2.legend()
# -------------------------------------------------------------------

fig1.savefig('Comparation_metrics_v3.svg',format='svg')
fig2.savefig('Comparation_loss_v3.svg',format='svg')

#plt.show()
# -------------------------------------------------------------------

for nn,mm in enumerate(metrics_table):
	for ee, kk in enumerate(exp_metrics):
		print(metrics_name[mm],kk,exp_metrics[kk][0,mm],exp_metrics[kk][-1,mm])

# -------------------------------------------------------------------

 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] exp_34 0.023 0.053
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] exp_35 0.034 0.101
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] exp_36 0.103 0.15
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] exp_37 0.138 0.27
 # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] exp_34 0.058 0.125
 # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] exp_35 0.08 0.21
 # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] exp_36 0.202 0.264
 # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] exp_37 0.265 0.465
 # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] exp_34 0.014 0.036
 # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] exp_35 0.023 0.085
 # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] exp_36 0.093 0.148
 # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] exp_37 0.134 0.276
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] exp_34 0.039 0.085
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] exp_35 0.063 0.137
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] exp_36 0.134 0.18
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] exp_37 0.16 0.245
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] exp_34 0.067 0.146
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] exp_35 0.106 0.233
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] exp_36 0.204 0.268
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] exp_37 0.241 0.391
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] exp_34 0.071 0.153
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] exp_35 0.109 0.243
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] exp_36 0.213 0.279
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] exp_37 0.248 0.411
