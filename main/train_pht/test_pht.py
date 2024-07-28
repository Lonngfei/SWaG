from glob import glob
from predict import predict
import os
import shutil

npy_pth = glob('../tf_psample/bht_test_swag_npy_100HZ_dxy16/*')
newdir = 'tf' 
if os.path.isdir(newdir):
    shutil.rmtree(newdir)
os.makedirs(newdir)
#model = '/home/dragonfly/Documents/Data_Processing/SWaG_BHT/train_pht_swag/results/002-PhaseNet/checkpoints/4350000.pt'
model1 = 'results/064-PhaseNet/checkpoints/0001500.pt'
for i in npy_pth:
    p_dic, s_dic = predict(i,  0.5, 0.5, model1)
    newfile = newdir + '/' + i.split('/')[-1][:-4]
    with open(newfile, 'w') as f:
        for p_time, p_prob in p_dic.items():
            f.write(f'{p_time} {p_prob} P\n')
        for s_times, s_prob in s_dic.items():
            f.write(f'{s_times} {s_prob} S\n')



