import matplotlib.pyplot as plt
import numpy as np
import argparse

PATH = '/home/osvaldo/Documents/CCNY/Project_RBP/results/classification/'
PATHfiles = PATH + 'training-evolution/'
PATHplots = PATH + 'plots/' 

parser = argparse.ArgumentParser(description='Comparation of experiments')
parser.add_argument('--exps', type=str, nargs='+', help='List of experiments')
parser.add_argument('--show', default=False, type=bool)
parser.add_argument('--save', default=False, type=bool)
args = parser.parse_args()
listexps = args.exps
num_exps = len(listexps)

# Plotting --------------------------------------------------
titles = ['Training','Validation']
ylabels = ['Loss function','% Correct prediction']


fig,axs = plt.subplots(2,2, figsize=(10,10))
for j in range(2):
	axs[1,j].set_ylim(0,100)
	axs[0,j].set_ylim(0,2.5)

	axs[0,j].set_title(titles[j])
	axs[1,j].set_xlabel('Epochs')
	axs[j,0].set_ylabel(ylabels[j])

# ----------------------------------------------------------	
string = '' 
for exp in listexps:
	data = np.loadtxt(PATHfiles + 'exp_' + exp +'.dat') # order: train_loss, train_acc, eval_loss, eval_acc

	for x in range(4):
		i = x%2
		j = int(x/2)

		axs[i,j].plot(data[0,:],data[x+1,:],label='exp_' + exp)
		axs[i,j].legend()
		axs[i,j].set_xlim(0,200)

	string = string + exp + '_'

if args.save:
	fig.savefig(PATHplots + 'training_exps_' + string + '.svg', format='svg')

if args.show:
	plt.show()