import matplotlib.pyplot as plt


dict_exp = {30: ([],'2LayerswithFeedBack'), 31: ([],'1LayerFeedForward'), 32: ([],'2LayersFeedForward')}
fig,axs = plt.subplots(1,1)

for exp in [30,31,32]:
	with open('/home/osvaldo/Documents/CCNY/Project_RBP/results_' + str(exp) + '_v22.txt') as f:
		lines = f.readlines()

		print(len(lines))

	for ii in range(10):
		dict_exp[exp][0].append(float(lines[2+19*ii][21:]))

	axs.plot(dict_exp[exp][0],label=dict_exp[exp][1])

axs.set_xlabel('Epochs')
axs.set_ylabel('Loss')
axs.legend()

plt.show()
plt.rcParams['svg.fonttype'] = 'none'

fig.savefig('Detection_Loops.svg', format='svg')