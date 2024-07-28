from PIL import Image
import torch
import torchvision as tv
from glob import glob
import numpy as np
import os
import natsort
from matplotlib import pyplot as plt


def create_thumbnail(dir, save_dir):
    image_list = []
    for image_path in natsort.natsorted(glob(os.path.join(dir, '*.png'))):
        img = Image.open(image_path)
        img = np.asarray(img)[..., :3] / 255
        image_list.append(img)
    image_list = np.stack(image_list)
    image_tensor = torch.from_numpy(image_list).permute(0, 3, 1, 2).float()
    tv.utils.save_image(image_tensor, os.path.join(save_dir, os.path.basename(os.path.dirname(dir)) + '_thumbnail.png'),
                        normalize=False, nrow=10)

def plot_npy(npy_dir, dir, fig_num):
    cash = 0
    for npy_path in glob(os.path.join(npy_dir, '*.npy')):
        cash += 1
        if cash == fig_num:
            break
        data_numpy = np.load(npy_path)
        label = npy_path.split('/')[-1][:-4]
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        station, p_sample, s_sample, mag = label.split('_')[:4]
        station = int(station)
        p_sample = float(p_sample)
        s_sample = float(s_sample)
        mag = float(mag)
        channels = ['E', 'N', 'Z']
        for i, ax in enumerate(axs):
            ax.plot(np.arange(0, 60, 0.02), data_numpy[:,i], c='black', linewidth=1.5)
            ax.set_ylabel(f"Channel {channels[i]}")
            ax.grid(True)
            ylim_min, ylim_max = ax.get_ylim()
            y_start = ylim_min + ((ylim_max - ylim_min) * (5/6))
            y_end = ylim_max - ((ylim_max - ylim_min) * (5/6))
            ax.plot([p_sample/50, p_sample/50], [y_start, y_end], 'b-', label='P', linewidth=2.5)
            ax.plot([s_sample/50, s_sample/50], [y_start, y_end], 'r-', label='S', linewidth=2.5)
            ax.set_xlim((p_sample/50) - 1, (s_sample/50) + ((mag+2)*2))
            #ax.set_xlim((p_sample/50) - 1, (s_sample/50) + 4)
            ax.legend(loc='upper right')
        fig.suptitle('Station:%d  P:%.4fs  S:%.4fs  Magnitude:%.2f'%(station, p_sample/50, s_sample/50, mag))
        axs[-1].set_xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, 'fig_' + str(cash) +'_' + label + '.png'))
        plt.close()



if __name__ == '__main__':
    save_dir = 'CSH'
    os.makedirs(save_dir, exist_ok=True)
    npy_dir = 'stead_CSH_CFG1.5_SampStep50_batchSample_500K'
    #npy_dir = '/D/Seismological_Data/STEAD/train_stead_data'
    dirs = 'L16A_Sample'
    os.makedirs(dirs, exist_ok=True)
    plot_npy(npy_dir, dirs, 101)
    #create_thumbnail(dirs, save_dir)
