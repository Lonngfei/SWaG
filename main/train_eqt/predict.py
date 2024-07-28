import torch
from eqtransformer import EQTransformer
import numpy as np
import os
from detect_peaks import detect_peaks

def predict(npy, det_prob, p_prob, s_prob, model):
    data = np.load(npy)
    data_tensor = torch.from_numpy(data).float().permute(1, 0)
    data_tensor = data_tensor.unsqueeze(0)
    weight  = torch.load(model)
    model = EQTransformer(in_samples=3000, sampling_rate=50, lstm_blocks=2)
    model.load_state_dict(weight)
    model.eval()
    p_pick = {}
    s_pick = {}
    with torch.no_grad():
        y = model(data_tensor)
        det, p, s = y
        det = det.numpy()
        p = p.numpy()
        s = s.numpy()
    p_idx_, p_prob_ = detect_peaks(p[0], mph=p_prob, mpd=50)
    s_idx, s_prob = detect_peaks(s[0], mph=s_prob, mpd=50)
    for i, j in zip(p_idx_, p_prob_):
        if det[0][i] >= det_prob:
            p_pick[i] = j
    for k, l in zip(s_idx, s_prob):
        if det[0][k] >= det_prob:
            s_pick[k] = l
    return p_pick, s_pick

if __name__ == '__main__':
    print(predict('1_200.0000_300.0000_0.40_00005811.npy', 0.5, 0.5, 0.5))
