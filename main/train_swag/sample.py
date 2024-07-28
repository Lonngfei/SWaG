# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os.path
import shutil

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT
import argparse


@torch.no_grad()
def main(args, ckpt_path, seed, depth, hidden_size, num_head, hidden_feature, length, num_class, cfg_scale, epochs, num_sampling_steps=250):
    # Setup PyTorch:
    #torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'

    # Load model:
    model = DiT(depth=depth, hidden_size=hidden_size, num_heads=num_head,
                hidden_feature=hidden_feature, length=length, learn_sigma=True)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage) ['ema']
    model.load_state_dict(state_dict)
    model.eval()  # important!
    model.half()
    model.to(device)
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Labels to condition the model with (feel free to change):
    sta_labels = torch.tensor(args.y_sta)
    sam_labels = (torch.tensor(args.y_sam)-args.p_sample_min)*(args.max_get-args.min_get)/(args.p_sample_max-args.p_sample_min)
    dis_labels = (torch.tensor(args.y_dis)-args.s_sample_min)*(args.max_get-args.min_get)/(args.s_sample_max-args.s_sample_min)
    mag_labels = (torch.tensor(args.y_mag)-args.mag_min)*(args.max_get-args.min_get)/(args.mag_max-args.mag_min)
    y = torch.stack([sta_labels, sam_labels, dis_labels, mag_labels], dim=1).to(device)



    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    if os.path.exists('SampledNumpy'):
        shutil.rmtree('SampledNumpy')
    os.mkdir('SampledNumpy')

    for e in tqdm(range(args.epochs)):
        # Create sampling noise:
        with torch.autocast(device_type="cuda"):
            z = torch.randn(args.n, 3, args.length,  device=device)
            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([5000, 0 ,0, 0])
            y_null = y_null.view(1, -1)
            y_null = y_null.repeat(args.n, 1).to(device)
            y_ = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_, cfg_scale=args.cfg_scale)
        
            samples = diffusion.p_sample_loop(model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device)
        samples = samples.detach().cpu().numpy()
        for j in range(int((samples.shape[0])/2)):
            p_sample = args.y_sam[j]
            s_sample = args.y_dis[j]
            mag = args.y_mag[j]
            channels = ['E', 'N', 'Z']
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(np.arange(0, 60, 0.02), samples[j, i, :], c='black', linewidth=1)
                ax.set_ylabel(f"Channel {channels[i]}")
                ax.grid(True)
                ylim_min, ylim_max = ax.get_ylim()
                y_start = ylim_min + ((ylim_max - ylim_min) * (5/6))
                y_end = ylim_max - ((ylim_max - ylim_min) * (5/6))
                ax.plot([p_sample/50, p_sample/50], [y_start, y_end], 'b-', label='P', linewidth=2)
                ax.plot([s_sample/50, s_sample/50], [y_start, y_end], 'r-', label='S', linewidth=2)
                ax.set_xlim((p_sample/50) - 1, (s_sample/50) + ((mag+2)*2))
                ax.legend(loc='upper right')
            axs[-1].set_xlabel('Time')
            station = args.y_sta[j]
            fig.suptitle('Station:%d  P:%.4fs  S:%.4fs  Magnitude:%.2f'%(station, p_sample/50, s_sample/50, mag))
            plt.tight_layout()
            plt.savefig(f'./{save_dir}/%d_%d.png' % (e, j))
            plt.close()
        savename = './SampledNumpy/%d.npy' % (e)
        np.save(savename, samples)
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--depth", default=24, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_heads", default=12, type=int)
    parser.add_argument("--hidden_feature", default=168, type=int)
    parser.add_argument("--ckpt", type=str, default="results/0100000.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--sample-dir", type=str, default="SampledImages_2.6Bck50k")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--t", type=int, default=1000)
    parser.add_argument("--station", type=int, default=1)
    parser.add_argument("--num_class", type=int, default=5000)
    parser.add_argument('--wf_dir', default = 'data/stead_swag_npy_50HZ', type = str)
    parser.add_argument("--p_sample_min", default=10, type=int)
    parser.add_argument("--p_sample_max", default=1500, type=int)
    parser.add_argument("--s_sample_min", default=50, type=int)
    parser.add_argument("--s_sample_max", default=2400, type=int)
    parser.add_argument("--mag_min", default=-1, type=int)
    parser.add_argument("--mag_max", default=8, type=int)
    parser.add_argument("--min_get", default = 0, type = int)
    parser.add_argument("--max_get", default = 1000, type = int)
    parser.add_argument("--length", default=3000, type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n", default=12, type=int)
    
    parser.add_argument('--y-sta', default=[92,      2414,    92,      227,     2414,    200,     200,     92,      1246,    92,      92,      25], type=int, nargs="+")
    parser.add_argument('--y-sam', default=[300.0000,250.0000,449.0000,200.0000,250.0000,200.0000,450.0000,400.0000,250.0000,399.5000,300.0000,400.0000], type=float, nargs="+")
    parser.add_argument('--y-dis', default=[791.0000,647.5000,601.0000,664.0000,640.5000,455.5000,707.0000,526.0000,391.0000,870.0000,374.0000,827.0000], type=float, nargs="+")
    parser.add_argument('--y-mag', default=[3.00,    1.00,    0.68,    1.10,    0.97,    2.21,    1.90,    0.90,    0.76,    1.39,    0.15,    1.10], type=float, nargs="+")
    parser.add_argument('--num_sampling_steps', default=1000, type=int)
    args = parser.parse_args()
    ckpt_path = args.ckpt
    cfg_scale = args.cfg_scale
    seed = 0
    length = args.length
    batch_size = 5
    epochs = 5
    log_every = 50
    depth = args.depth
    hidden_size = args.hidden_size
    num_head = args.num_heads
    num_class = args.num_class
    hidden_feature = 168
    station = args.station
    sample_dir = args.sample_dir
    num_sampling_steps = args.num_sampling_steps
    save_dir = f"{sample_dir}_{cfg_scale}_{num_sampling_steps}"
    import time
    time0 = time.time()
    main(args, ckpt_path, seed, depth, hidden_size, num_head, hidden_feature, length, num_class, cfg_scale,
         epochs)
    time1 = time.time()
    print(f'comsumed time:{int((time1-time0))}s')
         
    from PIL import Image
    import torch
    import torchvision as tv
    from glob import glob
    import numpy as np
    import os
    import natsort
    
    def create_thumbnail(dir, save_dir2, time):
        image_list = []
        for image_path in natsort.natsorted(glob(os.path.join(dir, '*.png'))):
            img = Image.open(image_path)
            img = np.asarray(img)[...,:3] / 255
            image_list.append(img)
        image_list = np.stack(image_list)
        image_tensor = torch.from_numpy(image_list).permute(0,3,1,2).float()
        tv.utils.save_image(image_tensor, os.path.join(save_dir2, f'{save_dir}_{int(time)}min_thumbnail.png'), normalize=False, nrow = 12)
    
    save_dir2 = './thumbnails'
    os.makedirs(save_dir2, exist_ok=True)
    
    dir = f"{save_dir}"
    create_thumbnail(dir, save_dir2, (time1-time0)/60)

