"""
This file runs the main training/val loop (corrected)
"""
import os
import sys
import pprint
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0  # default for single GPU

def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.images_dataset import ImagesDataset
from models.age import AGE
from options.train_options import TrainOptions
from criteria import orthogonal_loss, sparse_loss
from criteria.lpips.lpips import LPIPS
from optimizer.ranger import Ranger
from utils import common, train_utils
import torch.multiprocessing as mp

def train():
    opts = TrainOptions().parse()

    # Initialize distributed only if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # single GPU or CPU
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() < 2:
            print("⚠️ Single GPU detected, running without torch.distributed.")

    opts.device = device

    # Setup directories & logger only on main process
    if local_rank == 0:
        print("opts.exp_dir:", opts.exp_dir)
        if os.path.exists(opts.exp_dir):
            raise Exception('Oops... {} already exists'.format(opts.exp_dir))
        os.makedirs(opts.exp_dir)

        opts_dict = vars(opts)
        pprint.pprint(opts_dict)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_loss = None
        if opts.save_interval is None:
            opts.save_interval = opts.max_steps
    else:
        logger = None
        checkpoint_dir = None
        best_val_loss = None

    # Initialize network
    net = AGE(opts).to(device)
    params = list(net.ax.parameters())
    if opts.optim_name == 'adam':
        optimizer = torch.optim.Adam(params, lr=opts.learning_rate)
    else:
        optimizer = Ranger(params, lr=opts.learning_rate)

    # Estimate latent_avg via dense sampling if latent_avg is not available
    if net.latent_avg is None:
        net.latent_avg = net.decoder.mean_latent(int(1e5))[0].detach()

    # Wrap with DDP only if multiple GPUs
    if torch.cuda.device_count() > 1:
        net = nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )

    # Initialize dataset
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = {
        'transform_gt_train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'transform_source': None,
        'transform_valid': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'transform_inference': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_dataset = ImagesDataset(
        source_root=dataset_args['train_source_root'],
        target_root=dataset_args['train_target_root'],
        opts=opts,
        source_transform=transforms_dict['transform_source'],
        target_transform=transforms_dict['transform_gt_train']
    )
    valid_dataset = ImagesDataset(
        source_root=dataset_args['valid_source_root'],
        target_root=dataset_args['valid_target_root'],
        opts=opts,
        source_transform=transforms_dict['transform_source'],
        target_transform=transforms_dict['transform_valid']
    )

    if local_rank == 0:
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of valid samples: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if torch.cuda.device_count() > 1 else None
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False, drop_last=True) if torch.cuda.device_count() > 1 else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        num_workers=int(opts.workers),
        sampler=train_sampler,
        shuffle=(train_sampler is None)
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=opts.valid_batch_size,
        num_workers=int(opts.valid_workers),
        sampler=valid_sampler,
        shuffle=(valid_sampler is None)
    )

    # Initialize loss modules (on device) if needed
    lpips = LPIPS(net_type='vgg').to(device).eval() if opts.lpips_lambda > 0 else None
    sparse = sparse_loss.SparseLoss().to(device).eval() if opts.sparse_lambda > 0 else None
    orthogonal = orthogonal_loss.OrthogonalLoss(opts).to(device).eval() if opts.orthogonal_lambda > 0 else None

    # Training loop
    global_step = 0
    done = False
    net.train()

    # Loop until we reach max steps. We break cleanly once condition met.
    while not done:
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            x, y, av_codes = batch
            x, y, av_codes = x.to(device).float(), y.to(device).float(), av_codes.to(device).float()
            outputs = net.forward(x, av_codes, return_latents=True)
            loss, loss_dict, id_logs = calc_loss(opts, outputs, y, orthogonal, sparse, lpips)
            loss.backward()
            optimizer.step()

            # Important: increment immediately after optimizer.step()
            global_step += 1

            # Reduce losses across ranks (multi-GPU) if needed
            loss_dict = reduce_loss_dict(loss_dict)

            # Logging (only main rank)
            if local_rank == 0:
                if global_step % opts.image_interval == 0 or (global_step < 1000 and global_step % 25 == 0):
                    # parse_and_log_images internally checks/get logger handling
                    parse_and_log_images(opts, logger, global_step, id_logs, x, y, outputs['y_hat'], title='images/train/faces')
                if global_step % opts.board_interval == 0:
                    print_metrics(global_step, loss_dict, prefix='train')
                    log_metrics(logger, global_step, loss_dict, prefix='train')

            # Validation
            val_loss_dict = None
            if (global_step % opts.val_interval == 0) or (global_step >= opts.max_steps):
                if local_rank == 0:
                    val_loss_dict = validate(opts, net, orthogonal, sparse, lpips, valid_dataloader, device, global_step, logger)
                    if val_loss_dict and (best_val_loss is None or val_loss_dict['loss'] < best_val_loss):
                        best_val_loss = val_loss_dict['loss']
                        # pass scalar best_val_loss
                        checkpoint_me(net, opts, checkpoint_dir, float(best_val_loss), global_step, loss_dict, is_best=True)
                else:
                    # other ranks still run validate but do not checkpoint
                    val_loss_dict = validate(opts, net, orthogonal, sparse, lpips, valid_dataloader, device, global_step)

            # Checkpointing (main rank only)
            if local_rank == 0 and ((global_step % opts.save_interval == 0) or (global_step >= opts.max_steps)):
                if val_loss_dict is not None:
                    # write scalar loss into timestamp
                    checkpoint_me(net, opts, checkpoint_dir, float(val_loss_dict.get('loss', float(loss_dict.get('loss', 0.0)))), global_step, loss_dict, is_best=False)
                else:
                    checkpoint_me(net, opts, checkpoint_dir, float(loss_dict.get('loss', 0.0)), global_step, loss_dict, is_best=False)

            # Final stop: when we've reached or exceeded max_steps, exit cleanly
            if global_step >= opts.max_steps:
                if local_rank == 0:
                    print('OMG, finished training!')
                done = True
                break

        # end for dataloader
        if done:
            break

def validate(opts, net, orthogonal, sparse, lpips, valid_dataloader, device, global_step, logger=None):
    net.eval()
    agg_loss_dict = []
    for batch_idx, batch in enumerate(valid_dataloader):
        x, y, av_codes = batch
        with torch.no_grad():
            x, y, av_codes = x.to(device).float(), y.to(device).float(), av_codes.to(device).float()
            outputs = net.forward(x, av_codes, return_latents=True)
            loss, cur_loss_dict, id_logs = calc_loss(opts, outputs, y, orthogonal, sparse, lpips)
        agg_loss_dict.append(cur_loss_dict)

        # Logging related - only on main rank or single GPU
        if get_rank() == 0 or torch.cuda.device_count() < 2:
            parse_and_log_images(opts, logger, global_step, id_logs, x, y, outputs['y_hat'], title='images/valid/faces', subscript='{:04d}'.format(batch_idx))

        # For first step just do sanity valid on small amount of data
        if global_step == 0 and batch_idx >= 4:
            net.train()
            return None  # Do not log, inaccurate in first batch

    net.train()
    loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
    loss_dict = reduce_loss_dict(loss_dict)

    # Only log on rank 0 or single GPU
    if get_rank() == 0 or torch.cuda.device_count() < 2:
        log_metrics(logger, global_step, loss_dict, prefix='valid')
        print_metrics(global_step, loss_dict, prefix='valid')

    return loss_dict

def checkpoint_me(net, opts, checkpoint_dir, best_val_loss, global_step, loss_dict, is_best):
    """
    Save model and append a timestamp line into timestamp.txt.
    best_val_loss should be a scalar (float) when provided for non-best checkpoints too.
    """
    save_name = 'best_model.pt' if is_best else f'iteration_{global_step}.pt'
    save_dict = __get_save_dict(net, opts)
    checkpoint_path = os.path.join(checkpoint_dir, save_name)
    torch.save(save_dict, checkpoint_path)
    with open(os.path.join(checkpoint_dir, 'timestamp.txt'), 'a') as f:
        if is_best:
            f.write(f'**Best**: Iteration - {global_step}, Loss - {best_val_loss} \n{loss_dict}\n')
        else:
            f.write(f'Iteration - {global_step}, Loss - {best_val_loss} \n{loss_dict}\n')

def calc_loss(opts, outputs, y, orthogonal, sparse, lpips):
    loss_dict = {}
    loss = 0.0
    id_logs = None
    if opts.l2_lambda > 0:
        loss_l2 = F.mse_loss(outputs['y_hat'], y)
        loss_dict['loss_l2'] = loss_l2
        loss += loss_l2 * opts.l2_lambda
    if opts.lpips_lambda > 0:
        loss_lpips = lpips(outputs['y_hat'], y)
        loss_dict['loss_lpips'] = loss_lpips
        loss += loss_lpips * opts.lpips_lambda
    if opts.orthogonal_lambda > 0:
        loss_orthogonal_AB = orthogonal(outputs['A'])
        loss_dict['loss_orthogona'] = loss_orthogonal_AB
        loss += (loss_orthogonal_AB) * opts.orthogonal_lambda
    if opts.sparse_lambda > 0:
        loss_l1 = sparse(outputs['x'])
        loss_dict['loss_sparse'] = loss_l1
        loss += loss_l1 * opts.sparse_lambda
    loss_dict['loss'] = loss
    return loss, loss_dict, id_logs

def log_metrics(logger, global_step, metrics_dict, prefix):
    if logger is None:
        return
    for key, value in metrics_dict.items():
        # ensure we pass a scalar float to tensorboard
        try:
            scalar_value = float(value)
        except Exception:
            try:
                scalar_value = float(value.item())
            except Exception:
                scalar_value = 0.0
        logger.add_scalar(f'{prefix}/{key}', scalar_value, global_step)

def print_metrics(global_step, metrics_dict, prefix):
    print(f'Metrics for {prefix}, step {global_step}')
    for key, value in metrics_dict.items():
        print(f'\t{key} = ', value)

def parse_and_log_images(opts, logger, global_step, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
    im_data = []
    batch_size = x.size(0)
        
    for i in range(min(display_count, batch_size)):  # never go out of bounds
        cur_im_data = {
            'input_face': common.log_input_image(x[i], opts),
            'target_face': common.tensor2im(y[i]),
            'output_face': common.tensor2im(y_hat[i]),
        }
        if id_logs is not None and i < len(id_logs):
            for key in id_logs[i]:
                cur_im_data[key] = id_logs[i][key]
        im_data.append(cur_im_data)
    log_images(logger, global_step, title, im_data=im_data, subscript=subscript)

def log_images(logger, global_step, name, im_data, subscript=None, log_latest=False):
    fig = common.vis_faces(im_data)
    step = global_step
    if log_latest:
        step = 0
    if subscript:
        path = os.path.join(logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
    else:
        path = os.path.join(logger.log_dir, name, f'{step:04d}.jpg')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def __get_save_dict(net, opts):
    # Handle both single GPU and multi-GPU cases
    if torch.cuda.device_count() > 1 and isinstance(net, nn.parallel.DistributedDataParallel):
        # Multi-GPU: use net.module
        state_dict = net.module.state_dict()
        if opts.start_from_latent_avg:
            latent_avg = net.module.latent_avg
    else:
        # Single GPU (or not DDP): use net directly
        # net might be the module already
        try:
            state_dict = net.state_dict()
        except Exception:
            state_dict = net.module.state_dict()
        if opts.start_from_latent_avg:
            latent_avg = getattr(net, 'latent_avg', None)

    save_dict = {
        'state_dict': state_dict,
        'opts': vars(opts)
    }

    # save the latent avg in save_dict for inference if truncation of w was used during training
    if opts.start_from_latent_avg and 'latent_avg' in locals() and locals()['latent_avg'] is not None:
        save_dict['latent_avg'] = locals()['latent_avg']

    return save_dict

def reduce_loss_dict(loss_dict):
    # Only use distributed reduction if we're actually using multiple GPUs
    if torch.cuda.device_count() < 2:
        # Convert tensors to python floats for consistent printing
        return {k: (float(v) if hasattr(v, 'item') or isinstance(v, (int, float)) else v) for k, v in loss_dict.items()}

    world_size = dist.get_world_size()

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if get_rank() == 0:
            losses /= world_size

        # On non-zero ranks the reduced values are placeholders; only rank 0 will get correct averaged numbers
        if get_rank() == 0:
            reduced_losses = {k: float(v) for k, v in zip(keys, losses)}
        else:
            reduced_losses = {k: float(loss_dict[k]) if hasattr(loss_dict[k], 'item') else float(loss_dict[k]) for k in keys}

    return reduced_losses

def configure_datasets(opts, local_rank):
    if opts.dataset_type not in data_configs.DATASETS.keys():
        Exception(f'{opts.dataset_type} is not a valid dataset_type')
    if local_rank==0 or torch.cuda.device_count() < 2:
        print(f'Loading dataset for {opts.dataset_type}')
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                  target_root=dataset_args['train_target_root'],
                                  average_code_root=opts.class_embedding_path,
                                  source_transform=transforms_dict['transform_source'],
                                  target_transform=transforms_dict['transform_gt_train'],
                                  opts=opts)
    valid_dataset = ImagesDataset(source_root=dataset_args['valid_source_root'],
                                 target_root=dataset_args['valid_target_root'],
                                 average_code_root=opts.class_embedding_path,
                                 source_transform=transforms_dict['transform_source'],
                                 target_transform=transforms_dict['transform_valid'],
                                 opts=opts)
    if local_rank==0 or torch.cuda.device_count() < 2:
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of valid samples: {len(valid_dataset)}")
    return train_dataset, valid_dataset

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    train()
