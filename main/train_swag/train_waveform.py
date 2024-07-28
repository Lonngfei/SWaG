import argparse

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import logging
import os
from models import DiT
from diffusion import create_diffusion
from data_load import LoadWaveform
from tqdm import tqdm
import encoding
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class Timer:
    def __init__(self, name, show = True):
        self.name = name
        self.show = show

    def __enter__(self):
        self.start_time = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time()
        execution_time = end_time - self.start_time
        formatted_time = "{:.4f}".format(execution_time)
        if self.show:
            print(f"{self.name} costs {formatted_time}s")
            
def requires_grad(model, flag = True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level = logging.INFO,
            format = '[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt = '%Y-%m-%d %H:%M:%S',
            handlers = [logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay = 0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha = 1 - decay)


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    # basic setting
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok = True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok = True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # create model:
    model = DiT(depth = args.depth, hidden_size = args.hidden_size, num_heads = args.num_heads,
                hidden_feature = args.hidden_feature, length = args.length, learn_sigma = True)
    if args.pretrained_model:
        state_dict = torch.load(args.pretrained_model, map_location=lambda storage, loc: storage) ['ema']
        model.load_state_dict(state_dict)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    # requires_grad(ema, False)
    model = DDP(model.to(device), device_ids = [rank], gradient_as_bucket_view = True)
    diffusion = create_diffusion(timestep_respacing = "")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0)

    # load data
    dataset = LoadWaveform(args.wf_dir, args.p_sample_min, args.p_sample_max, args.s_sample_min, args.s_sample_max,
                           args.mag_min, args.mag_max, args.min_get, args.max_get)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info(f"Dataset contains {len(dataset):,} waveform ({args.wf_dir})")

    log_every = int(len(dataset) / args.global_batch_size) # args.log_every  
    # Prepare models for training:
    update_ema(ema, model.module, decay = 0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in tqdm(loader):

            x = x.to(device)
            y = y.to(device)
            # with torch.no_grad():
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device = device)
            model_kwargs = dict(y = y)
            # import pdb;pdb.set_trace()
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device = device)
                dist.all_reduce(avg_loss, op = dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--global-batch-size", default = 256, type = int)
    parser.add_argument("--global_seed", default = 0, type = int)
    parser.add_argument("--model", type = str, default = "SWaG")
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument('--is_transfer', default=True, type= bool)
    parser.add_argument("--results_dir", default = 'results', type = str)
    parser.add_argument('--wf_dir', default = 'data/stead_swag_npy_50HZ', type = str)
    parser.add_argument("--p_sample_min", default=10, type=int)
    parser.add_argument("--p_sample_max", default=1500, type=int)
    parser.add_argument("--s_sample_min", default=50, type=int)
    parser.add_argument("--s_sample_max", default=2400, type=int)
    parser.add_argument("--mag_min", default=-1, type=int)
    parser.add_argument("--mag_max", default=8, type=int)
    parser.add_argument("--min_get", default = 0, type = int)
    parser.add_argument("--max_get", default = 1000, type = int)
    parser.add_argument("--length", default = 3000, type = int)
    parser.add_argument("--num_workers", default = 12, type = int)
    parser.add_argument("--epochs", default = 200, type = int)
    parser.add_argument("--depth", default = 24, type = int)
    parser.add_argument("--hidden_size", default = 768, type = int)
    parser.add_argument("--num_heads", default = 12, type = int)
    parser.add_argument("--hidden_feature", default = 168, type = int)
    parser.add_argument("--ckpt-every", type = int, default = 50_000)
    parser.add_argument("--log-every", type = int, default = 100)
    args = parser.parse_args()
    main(args)

