from glob import glob
from predict import predict
import os
import shutil

npy_pth = glob('../tf_psample/bht_test_swag_npy_50HZ_dxy16/*')
newdir = 'bht_test_dataset' 
if os.path.isdir(newdir):
    shutil.rmtree(newdir)
os.makedirs(newdir)
model = 'results/165-EQTransformer/checkpoints/0125000.pt'
for i in npy_pth:
    p_dic, s_dic = predict(i, 0.1, 0.5, 0.5, model)
    print(i)
    newfile = newdir + '/' + i.split('/')[-1][:-4]
    with open(newfile, 'w') as f:
        for p_time, p_prob in p_dic.items():
            f.write(f'{p_time} {p_prob} P\n')
        for s_times, s_prob in s_dic.items():
            f.write(f'{s_times} {s_prob} S\n')



