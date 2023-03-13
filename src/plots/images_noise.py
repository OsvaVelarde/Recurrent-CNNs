from torchvision.datasets import CIFAR100, CIFAR10
import torchvision.transforms as tt

import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------

from noise_functions import pixel_noise,blurr_noise,contr_noise,occlu_noise

# -----------------------------------------------
num_channels=3

normalization = tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

num_classes = 10
test_data = CIFAR10(root='./data',train=False,transform=tt.ToTensor())
test_dataloader = DataLoader(test_data, batch_size = 1, shuffle=False, num_workers=2)
dataiter = iter(test_dataloader)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# -----------------------------------------------

fig_pxl,axs_pxl = plt.subplots(10,6,figsize=(40,20))
fig_blu,axs_blu = plt.subplots(10,6,figsize=(40,20))
fig_con,axs_con = plt.subplots(10,6,figsize=(40,20))
fig_occ,axs_occ = plt.subplots(10,6,figsize=(40,20))

noise_level = [0.25,0.5,0.75,1]
blurrer_sigma = [1,5,10,20]
contrast_factor = [0.8,0.5,0.2,0.1]
occlu_scale = [0.3,0.4,0.5,0.6]

dict_plots = {'pxl':(axs_pxl,pixel_noise,noise_level),
              'blurrer':(axs_blu,blurr_noise,blurrer_sigma),
              'contrast':(axs_con,contr_noise,contrast_factor),
              'occluded':(axs_occ,occlu_noise,occlu_scale)}

# -----------------------------------------------

for i in range(10):
    img, label = dataiter.next()
    img_norm = normalization(img)

    for key,values in dict_plots.items():
        values[0][i,0].imshow(np.transpose(img[0].numpy(), (1, 2, 0)))
        values[0][i,1].imshow(np.transpose(img_norm[0].numpy(), (1, 2, 0)))

        j = 2
        for nn in values[2]:
            transformed = values[1](img_norm[0],nn).numpy()
            values[0][i,j].imshow(np.transpose(transformed, (1, 2, 0)))
            j = j+1

plt.show()

# ---------------------------------------------------

