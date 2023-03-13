import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, top_k_accuracy_score

PATH = '/home/osvaldo/Documents/CCNY/Project_RBP/results/'

parser = argparse.ArgumentParser(description='Matrix Confusion')
parser.add_argument('--exp', type=str, required=True, help='Name of experiment')
parser.add_argument('--folder', default="predictions", choices=['predictions', 'evaluations'])
args = parser.parse_args()

prefix = 'exp_' if args.folder == 'predictions' else 'ev_'
filename = PATH + args.folder + '/' + prefix + args.exp + '.csv'

# Load data --------------------------------------
df = pd.read_csv(filename)
num_classes = 10
max_top_k = 3

# Confusion Matrix ------------------------------
cf_mtx = confusion_matrix(df['label'], df['prediction'])
correct_prediction = cf_mtx.trace()

# Top-K accuracies ------------------------------
for kk in range(max_top_k):
	k_acc = top_k_accuracy_score(df['label'], df[['class_'+str(ii) for ii in range(num_classes)]], k=kk+1, labels = [float(ii) for ii in range(10)])
	print('top_' + str(kk+1) + '_acc:',k_acc)

# Plotting --------------------------------------
fig, axs = plt.subplots(figsize=(7.5, 7.5))
axs.matshow(cf_mtx, cmap=plt.cm.YlOrRd, alpha=0.5)

for m in range(cf_mtx.shape[0]):
    for n in range(cf_mtx.shape[1]):
        axs.text(x=m,y=n,s=cf_mtx[m, n], va='center', ha='center', size='xx-large')

axs.set_xticks(range(0,10))
axs.set_yticks(range(0,10))

# Sets the labels
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Labels', fontsize=16)
plt.title('Confusion Matrix', fontsize=15)

fig.savefig(PATH + 'plots/' + prefix + args.exp + '.svg',format='svg',transparent=True)

#plt.show()


