import os
import numpy as np
import torch
from torch.utils.data import Dataset
from label import generate_label

class LoadWaveform(Dataset):
    def __init__(self, wf_dir):
        self.wf_dir = wf_dir
        self.wf_path = os.listdir(self.wf_dir)

    def __getitem__(self, idx):
        wf_name = self.wf_path[idx]
        wf_item_path = os.path.join(self.wf_dir, wf_name)
        wf = np.load(wf_item_path)
        wf_tensor = torch.from_numpy(wf).float().permute(1, 0)
        code, p_sample, s_sample, mag = wf_name.split('_')[:4]
        det = torch.zeros(3000)
        det_index = [i for i in range(int(float(p_sample)), int(float(s_sample) + 1.4 * (float(s_sample)-float(p_sample))))]
        det[det_index] = 1
        p = generate_label(3000, p_sample,"gaussian", 30)
        s = generate_label(3000, s_sample, "gaussian", 50)
        p = torch.from_numpy(p).float()
        s = torch.from_numpy(s).float()
        #label = torch.stack((det, p, s), 0)
        label = (det, p, s)
        return wf_tensor, label

    def __len__(self):
        return len(self.wf_path)
