from sklearn.metrics import top_k_accuracy_score
import numpy as np
import pandas as pd

def top_k(target,probs,num_classes,k=1):
	return top_k_accuracy_score(target, probs, k=k, labels = [float(ii) for ii in range(num_classes)])

def statistic_top_k(target,probs,num_classes,k=1,n_bootstraps=10,seed=42):

	bootstrapped_scores = []
	rng = np.random.RandomState(seed)

	for i in range(n_bootstraps):	
		indices = rng.randint(0, len(probs), len(probs))

		if len(np.unique(target[indices])) < 2:
			continue

		score = top_k(target.loc[indices], probs.loc[indices],num_classes,k)
		bootstrapped_scores.append(score)

	sorted_scores = np.array(bootstrapped_scores)
	sorted_scores.sort()
	std = np.std(sorted_scores)
	confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
	confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

	k_acc = top_k(target, probs, num_classes, k)

	return k_acc, std, sorted_scores, confidence_lower ,confidence_upper 

def plot_exp_noising(axs,k_acc,error,type_noise,noise_cfg,plt_cfg=None):
	name, xmin, xmax, dx  = noise_cfg
	fontsize = plt_cfg['fontsize']
	label = plt_cfg['label']
	x_range = np.arange(xmin,xmax+dx,dx)
	axs.set_xlim(xmin, xmax)
	axs.set_ylim(0,1)
	axs.set_xlabel('Level Noise', fontsize=fontsize)
	axs.set_ylabel('% Correct Prediction', fontsize=fontsize)
	axs.errorbar(x_range,k_acc,yerr=error,label=label)
	axs.legend(fontsize=fontsize)

def comparation_exp_noising(path, listexps, top_k, num_classes, num_files ,type_noise, noise_cfg, axs):
	pltfname = 'comparation_topk_'+ str(top_k) + '_' + type_noise + '_noise_exps_'
	k_acc_exps = {ee: [] for ee in listexps}
	k_acc_error_exps = {ee: [] for ee in listexps}

	for exp in listexps:
		plt_cfg = {'fontsize':12,'label':'exp_'+exp} 
		path_exp_noise = path + 'exp_' + exp + '/' + type_noise + '_noise/'
		for idx_ev in range(0,num_files):
			filename = path_exp_noise + 'ev_' + '%02d' % idx_ev + '.csv'
			df = pd.read_csv(filename)
			k_acc, k_std, _, _, _ = statistic_top_k(df['label'], df[['class_'+str(ii) for ii in range(num_classes)]], num_classes,top_k)
			k_acc_exps[exp].append(k_acc)
			k_acc_error_exps[exp].append(k_std) 

		plot_exp_noising(axs,k_acc_exps[exp],k_acc_error_exps[exp],type_noise,noise_cfg,plt_cfg)
		pltfname+= exp+'_'

	return pltfname