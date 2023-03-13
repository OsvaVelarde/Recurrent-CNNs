# Plot Exps vs Noise

# ===========================================================
# ============== MODULES & PATHS ============================
import matplotlib.pyplot as plt
import argparse
from plot_metrics import comparation_exp_noising

PATH = '/home/osvaldo/Documents/CCNY/Project_RBP/results/classification/'
PATHfiles = PATH + 'evaluations/'
PATHplots = PATH + 'plots/' 

# ===========================================================
# ============== IO & MODEL FILES ===========================

parser = argparse.ArgumentParser(description='Description')
parser.add_argument('--exps', nargs="+", default=[], type=str)
parser.add_argument('--type-noise', required=True, type=str)
parser.add_argument('--max-level', required=True, type=float)
parser.add_argument('--top-metric', required=True, type=int)
parser.add_argument('--dataset', required=True, type=int)
parser.add_argument('--num-files', required=True, type=int)

args = parser.parse_args()

num_files = args.num_files
num_classes = args.dataset
max_level = args.max_level
dlevel = max_level/(num_files-1)

OPTS_NOISES = {'pxl':('Additive Gaussian Pixel Noise',0,max_level,dlevel),
			   'occlu':('Occlusion',0,max_level,dlevel),
			   'blurr':('Gaussian Blur',0,max_level,dlevel)}

# ===========================================================
# ==================== PLOTTING =============================
fig,axs = plt.subplots(1,1, figsize=(10,10))
name = comparation_exp_noising(PATHfiles, args.exps, args.top_metric, num_classes, num_files,args.type_noise, OPTS_NOISES[args.type_noise], axs)
fig.savefig(PATHplots + name+'.svg',format='svg')
