import torch
from phasenet import PhaseNet, VariableLengthPhaseNet
import numpy as np
import os
from detect_peaks import detect_peaks
import matplotlib.pyplot as plt
def predict(npy, p_prob, s_prob, model):
    data = np.load(npy)
    data_tensor = torch.from_numpy(data).float().permute(1, 0)[:3000]
    data_tensor = data_tensor.unsqueeze(0)
    weight  = torch.load(model)
    #model = VariableLengthPhaseNet(in_samples=1500, sampling_rate=50)
    model = PhaseNet(sampling_rate=100)
    model.load_state_dict(weight)
    model.eval()
    p_pick = {}
    s_pick = {}
    with torch.no_grad():
        outputs = model(data_tensor)
    #fig = plt.figure(figsize=(15, 10))
    #axs = fig.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1, 1]})
    output_d, output_p, output_s = outputs[:,0,:].detach().cpu().numpy(), outputs[:,1,:].detach().cpu().numpy(), outputs[:,2,:].detach().cpu().numpy()
    #axs[0].plot(data_tensor[0].T)
    #axs[1].plot(output_d.T, c='black', label='P')
    #axs[1].plot(output_p.T, c='blue', label='S')
    #axs[1].plot(output_s.T, c='red', label='S')
    #axs[2].plot(sample[1][0].numpy().T, c='black', label='P')
    #axs[2].plot(sample[1][1].numpy().T, c='blue', label='S')
    #axs[2].plot(sample[1][2].numpy().T, c='red', label='S')
    #plt.legend()
    #plt.show()
    #plt.close()
    p_idx_, p_prob_ = detect_peaks(output_p[0], mph=p_prob, mpd=100)
    s_idx, s_prob = detect_peaks(output_s[0], mph=s_prob, mpd=100)
    for i, j in zip(p_idx_, p_prob_):
        #if det[0][i] >= det_prob:
        if True:
            p_pick[i] = j
    for k, l in zip(s_idx, s_prob):
        #if det[0][k] >= det_prob:
        if True:
            s_pick[k] = l
    return p_pick, s_pick

if __name__ == '__main__':
    print(predict('1_200.0000_300.0000_0.40_00005811.npy', 0.5, 0.5, 0.5))
