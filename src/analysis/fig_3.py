import matplotlib.pyplot as plt
from plot_metrics import comparation_exp_noising

PATH = '/home/osvaldo/Documents/CCNY/Project_RBP/results/'
PATHfiles = PATH + 'evaluations/'
PATHplots = PATH + 'plots/' 

# ===========================================================
OPTS_NOISES = {'pxl':('Additive Gaussian Pixel Noise',0,0.2,0.02),
			   'occlu':('Occlusion',0,1,0.1),
			   'blurr':('Gaussian Blur',0,1,0.1)}

EXPS_1L = ['04','05']
EXPS_2L = {'wo Feedback':['03','06','08'] , 'w Feedback':['03','07','09']}

top_metric = 1
num_files = 11
num_classes = 10

# ===========================================================
# ==================== PLOTTING =============================
fig_1,axs_1 = plt.subplots(1,3, figsize=(15,5))
fig_2,axs_2 = plt.subplots(2,3, figsize=(15,10))

for nn, type_noise in enumerate(OPTS_NOISES):
	# Plot 1 Layer Networks
	comparation_exp_noising(PATHfiles, EXPS_1L, top_metric, num_classes, num_files, type_noise, OPTS_NOISES[type_noise], axs_1[nn])

	# Plot 2 Layer Networks
	for ll, listexps in enumerate(EXPS_2L):
		comparation_exp_noising(PATHfiles, EXPS_2L[listexps], top_metric, num_classes, num_files, type_noise, OPTS_NOISES[type_noise], axs_2[ll,nn])

fig_1.savefig('TOP_1_Noises_1L.svg',format='svg')
fig_2.savefig('TOP_1_Noises_2L.svg',format='svg')