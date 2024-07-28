import argparse
import os
from time import time
from glob import glob
import torch.nn.init as init
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from data_load import LoadWaveform
import torch
from eqtransformer import EQTransformer
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if True:  # real logger
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

# def loss_fn(y_pred, y_true, eps=1e-5):
#     # vector cross entropy loss
#     h = y_true * torch.log(y_pred + eps)
#     h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
#     h = h.mean()  # Mean over batch axis
#     return -h

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def main(args):
    # basic setting
    rank = 0
    world_size = 1
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)
    #assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    #rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    #seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(args.global_seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={args.global_seed}, world_size={world_size}.")
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

    ## load model and data
    weight = torch.load('results/151-EQTransformer/checkpoints/1037500.pt')
    #model = EQTransformer(original_compatible=args.original_compatible, lstm_blocks=args.lstm_blocks).to(device)
    model = EQTransformer(in_samples=3000, sampling_rate=50, lstm_blocks=args.lstm_blocks).to(device)
    model.load_state_dict(weight)
    logger.info(f"EQTranformer Parameters: {sum(p.numel() for p in model.parameters()):,}")


    # init
    # for m in model.modules():
    #     if isinstance(m, nn.Linear):
    #         init.xavier_uniform_(m.weight)
    #         init.zeros_(m.bias)

    ## opt
    loss_func_d = nn.BCELoss()
    loss_func_p = nn.BCELoss()
    loss_func_s = nn.BCELoss()

    epochs = [0, 20, 40, 60, 90]
    lrs = [0.001, 0.1, 0.01, 0.001, 0.0005]

    lr_dict = {epoch: lr for epoch, lr in zip(epochs, lrs)}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[0])

    def lr_lambda_func(current_epoch):
        return lr_dict.get(current_epoch, lr_dict[max(lr_dict.keys())])
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_func)

    # load data
    train_dataset = LoadWaveform(args.train_dir)
    eval_dataset =  LoadWaveform(args.eval_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    logger.info(f"Train dataset contains {len(train_dataset)} waveform ({args.train_dir})")
    logger.info(f"Eval dataset contains {len(eval_dataset)} waveform ({args.eval_dir})")

    log_steps = 0
    train_loss = 0
    eval_loss = 0
    train_steps = 0
    eval_steps = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        model.train()
        for wf, labels in tqdm(train_loader):
            wf = wf.to(device)
            output_d, output_p, output_s = model(wf)
            label_d, label_p, label_s = labels[0].to(device), labels[1].to(device), labels[2].to(device)
            loss_d = loss_func_d(output_d, label_d)
            loss_p = loss_func_p(output_p, label_p)
            loss_s = loss_func_s(output_s, label_s)
            loss = 0.2*loss_d + 0.3*loss_p + 0.5*loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            log_steps += 1
            train_steps += 1
            train_loss += loss.item()


        model.eval()
        for eval, eval_labels in tqdm(eval_loader):
            eval = eval.to(device)
            with torch.no_grad():
                eval = eval.to(device)
                output_d, output_p, output_s = model(eval)
                label_d, label_p, label_s = eval_labels[0].to(device), eval_labels[1].to(device), eval_labels[2].to(device)
                loss_d = loss_func_d(output_d, label_d)
                loss_p = loss_func_p(output_p, label_p)
                loss_s = loss_func_s(output_s, label_s)
                loss = 0.2*loss_d + 0.3*loss_p + 0.5*loss_s

            eval_steps += 1
            eval_loss += loss.item()

        # sample = train_dataset[np.random.randint(len(train_dataset))]
        # fig = plt.figure(figsize=(15, 10))
        # axs = fig.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1, 1]})
        # outputs = model(sample[0].unsqueeze(0).to(device))
        # output_d, output_p, output_s = outputs[0].detach().cpu().numpy(), outputs[1].detach().cpu().numpy(), outputs[2].detach().cpu().numpy()
        # axs[0].plot(sample[0].T)
        # axs[1].plot(output_d.T, c='black', label='Detction')
        # axs[1].plot(output_p.T, c='blue', label='P')
        # axs[1].plot(output_s.T, c='red', label='S')
        # axs[2].plot(sample[1][0].numpy().T, c='black', label='Detction')
        # axs[2].plot(sample[1][1].numpy().T, c='blue', label='P')
        # axs[2].plot(sample[1][2].numpy().T, c='red', label='S')
        # plt.legend()
        # plt.show()
        # plt.close()

        if log_steps % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            avg_train_loss = torch.tensor(train_loss / train_steps, device=device)
            avg_eval_loss = torch.tensor(eval_loss / eval_steps, device=device)

            logger.info(
                f"(step={log_steps:07d}) Train Loss: {avg_train_loss:.8f}, Eval Loss: {avg_eval_loss:.8f} Train Steps/Sec: {steps_per_sec:.2f} ")
            # Reset monitoring variables:
            train_steps = 0
            eval_steps = 0
            train_loss = 0
            eval_loss = 0
            start_time = time()

        # Save eqt checkpoint:
        if log_steps % args.ckpt_every == 0 and log_steps > 0:
            if rank == 0:
                checkpoint_path = f"{checkpoint_dir}/{log_steps:07d}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--global_seed", default=0, type=int)
    parser.add_argument("--model", type=str, default="EQTransformer")
    parser.add_argument("--pretrained_model", type=str, default='ModelsAndSampleData/original2022.pt')
    parser.add_argument("--results_dir", default='results', type=str)
    parser.add_argument("--original_compatible", default='conservative', type=str)
    parser.add_argument("--lstm_blocks", default=2, type=int)
    parser.add_argument('--train_dir', default='/home/dragonfly/Documents/Data_Processing/SWaG_BHT/batch_generate/BHT_Size5000_CFG1.5_SampStep50_batchSample_500K', type=str)
    parser.add_argument('--eval_dir',
                        default='/home/dragonfly/Documents/Data_Processing/SWaG_BHT/tf_psample/bht_train_swag_npy_50HZ_dxy16',
                        type=str)
    parser.add_argument("--p_sample_min", default=100, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-1)
    args = parser.parse_args()
    main(args)
