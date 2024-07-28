import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LoadWaveform(Dataset):
    def __init__(self, wf_dir, p_sample_min, p_sample_max, s_sample_min, s_sample_max, mag_min, mag_max, min_get, max_get):
        self.wf_dir = wf_dir
        self.wf_path = os.listdir(self.wf_dir)
        self.p_sample_min = p_sample_min
        self.p_sample_max = p_sample_max
        self.s_sample_min = s_sample_min
        self.s_sample_max = s_sample_max
        self.mag_min = mag_min
        self.mag_max = mag_max
        self.min_get = min_get
        self.max_get = max_get

    def convert_to_zero_one(self, s, max_num, min_num):
        s = (float(s) - min_num) / (max_num - min_num)
        s = s * (self.max_get - self.min_get) + self.min_get
        return s

    def __getitem__(self, idx):
        wf_name = self.wf_path[idx]
        wf_item_path = os.path.join(self.wf_dir, wf_name)
        wf = np.load(wf_item_path)
        wf_tensor = torch.from_numpy(wf).float().permute(1, 0)
        code, p_sample, s_sample, mag = wf_name.split('_')[:4]
        label = torch.tensor([int(code), (self.convert_to_zero_one(float(p_sample), self.p_sample_max, self.p_sample_min)),
                                                 (self.convert_to_zero_one(float(s_sample), self.s_sample_max, self.s_sample_min)),
                                                 (self.convert_to_zero_one(float(mag), self.mag_max, self.mag_min))]).float()
        return wf_tensor, label

    def __len__(self):
        return len(self.wf_path)

if __name__ == '__main__':
    wf_dir = '../train_stead_data'
    p_sample_min, p_sample_max = 0, 800
    s_sample_min, s_sample_max = 40, 1200
    mag_min, mag_max = -1, 8
    min_get, max_get = 0, 1000
    dataset = LoadWaveform(wf_dir, p_sample_min, p_sample_max, s_sample_min, s_sample_max, mag_min, mag_max, min_get, max_get)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True, pin_memory=True)
