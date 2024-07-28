import math
from typing import Union
from data_load import LoadWaveform
import argparse
import os
from time import time
from glob import glob
import torch.nn.init as init
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import argparse
import os
from time import time
from glob import glob
import torch.nn.init as init
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from data_load import LoadWaveform
import torch
from phasenet import PhaseNet,VariableLengthPhaseNet
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR


train_dir = '/home/dragonfly/Documents/Data_Processing/SWaG_BHT/batch_generate/BHT_Size5000_CFG1.5_SampStep50_batchSample_500K'
eval_dir = '/home/dragonfly/Documents/Data_Processing/SWaG_BHT/tf_psample/bht_train_swag_npy_50HZ_dxy16'
test_dir = '/home/dragonfly/Documents/Data_Processing/SWaG_BHT/tf_psample/bht_test_swag_npy_50HZ_dxy16'
# load model
device = 'cpu'
weight = torch.load('results/059-PhaseNet/checkpoints/0022000.pt')
model = VariableLengthPhaseNet(in_samples=1500).to(device)
model.load_state_dict(weight)


#load data
train_dataset = LoadWaveform(train_dir)
eval_dataset = LoadWaveform(eval_dir)
test_dataset = LoadWaveform(test_dir)

# sample = train_dataset[np.random.randint(len(train_dataset))]
# fig = plt.figure(figsize=(15, 10))
# axs = fig.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1, 1]})
# outputs = model(sample[0].unsqueeze(0).to(device))
# output_d, output_p, output_s = outputs[0].detach().cpu().numpy(), outputs[1].detach().cpu().numpy(), outputs[2].detach().cpu().numpy()
# axs[0].plot(sample[0].T)
# axs[1].plot(output_d.T, c='black', label='Detction')
# axs[1].plot(output_p.T, c='blue', label='P')
# axs[1].plot(output_s.T, c='red', label='S')
# axs[2].plot(sample[1][0].numpy().T, c='black', label='Detction')
# axs[2].plot(sample[1][1].numpy().T, c='blue', label='P')
# axs[2].plot(sample[1][2].numpy().T, c='red', label='S')
# plt.legend()
# plt.show()
# plt.close()

sample = eval_dataset[np.random.randint(len(eval_dataset))]
fig = plt.figure(figsize=(15, 10))
axs = fig.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1, 1]})
outputs = model(sample[0].unsqueeze(0).to(device))
output_d, output_p, output_s = outputs[:,0,:].detach().cpu().numpy(), outputs[:,1,:].detach().cpu().numpy(), outputs[:,2,:].detach().cpu().numpy()
axs[0].plot(sample[0].T)
axs[1].plot(output_d.T, c='black', label='Detction')
axs[1].plot(output_p.T, c='blue', label='P')
axs[1].plot(output_s.T, c='red', label='S')
axs[2].plot(sample[1][0].numpy().T, c='black', label='Detction')
axs[2].plot(sample[1][1].numpy().T, c='blue', label='P')
axs[2].plot(sample[1][2].numpy().T, c='red', label='S')
plt.legend()
plt.show()
plt.close()


sample = test_dataset[np.random.randint(len(test_dataset))]
fig = plt.figure(figsize=(15, 10))
axs = fig.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1, 1]})
outputs = model(sample[0].unsqueeze(0).to(device))
output_d, output_p, output_s = outputs[:,0,:].detach().cpu().numpy(), outputs[:,1,:].detach().cpu().numpy(), outputs[:,2,:].detach().cpu().numpy()
axs[0].plot(sample[0].T)
axs[1].plot(output_d.T, c='black', label='P')
axs[1].plot(output_p.T, c='blue', label='S')
#axs[1].plot(output_s.T, c='red', label='S')
axs[2].plot(sample[1][0].numpy().T, c='black', label='P')
axs[2].plot(sample[1][1].numpy().T, c='blue', label='S')
#axs[2].plot(sample[1][2].numpy().T, c='red', label='S')
plt.legend()
plt.show()
plt.close()

