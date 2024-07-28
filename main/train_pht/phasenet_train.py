import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding
from data_load import LoadWaveform
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from seisbench.util import worker_seeding

model = sbm.PhaseNet(phases="PSN", norm="peak")

model.cuda()
batch_size = 50
num_workers = 4  # The number of threads used for loading data

learning_rate = 1e-2
epochs = 5

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train_dataset = LoadWaveform('/home/dragonfly/Documents/Data_Processing/SWaG_BHT/batch_generate/BHT_Size5000_CFG1.5_SampStep50_batchSample_500K')
eval_dataset = LoadWaveform('/home/dragonfly/Documents/Data_Processing/SWaG_BHT/tf_psample/bht_train_swag_npy_50HZ_dxy16')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,  worker_init_fn=worker_seeding,
                          num_workers=num_workers)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,  worker_init_fn=worker_seeding,
                         num_workers=num_workers)
def loss_fn(y_pred, y_true, eps=1e-5):
    # vector cross entropy loss
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h
def train_loop(dataloader):
    size = len(dataloader.dataset)
    for batch_id, batch in enumerate(dataloader):
        # Compute prediction and
        pred = model(batch[0].to(model.device))
        loss = loss_fn(pred, batch[1].to(model.device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch_id % 5 == 0:
        #     loss, current = loss.item(), batch_id * batch[0].shape[0]
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader):
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()  # close the model for evaluation

    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch[0].to(model.device))
            test_loss += loss_fn(pred, batch[1].to(model.device)).item()

    model.train()  # re-open model for training stage

    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader)
    test_loop(eval_loader)
torch.save(model.state_dict(), 'swag_phasenet.pth')