"""
Training script for SAGE using AGEWithFSE (FSE encoder + Ax module)
Single-GPU version, supports class_embedding
"""
import os
import sys
import pprint
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.images_dataset import ImagesDataset
from models.age_with_fse import AGEWithFSE
from options.train_options import TrainOptions
from criteria import orthogonal_loss, sparse_loss
from criteria.lpips.lpips import LPIPS
from optimizer.ranger import Ranger
from utils import common, train_utils


def train():
    opts = TrainOptions().parse()

    # Single-GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts.device = device
    local_rank = 0  # not used in single-GPU

    # Create experiment directories
    os.makedirs(opts.exp_dir, exist_ok=True)
    log_dir = os.path.join(opts.exp_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    pprint.pprint(vars(opts))
    logger = SummaryWriter(log_dir=log_dir)
    best_val_loss = None

    # Initialize network
    net = AGEWithFSE(opts).to(device)

    # Ensure latent_avg exists for FSE training
    if not hasattr(net, 'latent_avg') or net.latent_avg is None:
        print("[INFO] Computing latent_avg for decoder (FSE mode)...")
        net.latent_avg = net.decoder.mean_latent(int(1e5))[0].detach()

    net.train()

    # Optimizer: always include Ax, optionally feature editor
    params = list(net.ax.parameters())
    if opts.train_feature_editor:
        if hasattr(net.encoder, 'encoder'):  # FSEFull
            params += list(net.encoder.encoder.parameters())
        else:  # FSEInverter: train fuser only
            params += [p for n, p in net.encoder.named_parameters() if 'fuser' in n]

    optimizer = Ranger(params, lr=opts.learning_rate) if opts.optim_name.lower() != 'adam' else torch.optim.Adam(params, lr=opts.learning_rate)

    # Dataset transforms using torchvision
    transforms_dict = {
        'transform_gt_train': transforms.Compose([
            transforms.Resize((opts.output_size, opts.output_size)),
            transforms.ToTensor(),
        ]),
        'transform_source': transforms.Compose([
            transforms.Resize((opts.output_size, opts.output_size)),
            transforms.ToTensor(),
        ]),
        'transform_valid': transforms.Compose([
            transforms.Resize((opts.output_size, opts.output_size)),
            transforms.ToTensor(),
        ]),
        'transform_inference': transforms.Compose([
            transforms.Resize((opts.output_size, opts.output_size)),
            transforms.ToTensor(),
        ]),
    }

    # Dataset
    dataset_args = data_configs.DATASETS[opts.dataset_type]

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

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=int(opts.workers))
    valid_loader = DataLoader(valid_dataset, batch_size=opts.valid_batch_size, shuffle=False, num_workers=int(opts.valid_workers))

    # Losses
    lpips_loss = LPIPS(net_type='vgg').to(device).eval() if opts.lpips_lambda > 0 else None
    sparse_loss_fn = sparse_loss.SparseLoss().to(device).eval() if opts.sparse_lambda > 0 else None
    orthogonal_loss_fn = orthogonal_loss.OrthogonalLoss(opts).to(device).eval() if opts.orthogonal_lambda > 0 else None

    global_step = 0

    while global_step < opts.max_steps:
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x, y, av_codes = batch
            x, y, av_codes = x.to(device).float(), y.to(device).float(), av_codes.to(device).float()

            # Feature scale schedule
            current_scale = min(1.0, global_step / opts.feature_scale_warmup) if opts.feature_scale_schedule else opts.feature_scale

            outputs = net.forward(x, av_codes, return_latents=True, feature_scale=current_scale)
            loss, loss_dict, id_logs = calc_loss(opts, outputs, y, orthogonal_loss_fn, sparse_loss_fn, lpips_loss)
            loss.backward()
            optimizer.step()

            # Logging
            if global_step % opts.image_interval == 0 or (global_step < 1000 and global_step % 25 == 0):
                parse_and_log_images(opts, logger, global_step, id_logs, x, y, outputs['y_hat'], title='images/train')

            if global_step % opts.board_interval == 0:
                print_metrics(global_step, loss_dict, prefix='train')
                log_metrics(logger, global_step, loss_dict, prefix='train')

            # Validation
            if global_step % opts.val_interval == 0 or global_step == opts.max_steps:
                val_loss_dict = validate(opts, net, orthogonal_loss_fn, sparse_loss_fn, lpips_loss, valid_loader, device, global_step, logger)

                if val_loss_dict is not None and (best_val_loss is None or val_loss_dict['loss'] < best_val_loss):
                    best_val_loss = val_loss_dict['loss']
                    checkpoint_me(net, opts, checkpoint_dir, best_val_loss, global_step, loss_dict, is_best=True)

            # Save checkpoints
            if global_step % opts.save_interval == 0 or global_step == opts.max_steps:
                checkpoint_me(net, opts, checkpoint_dir, loss_dict, global_step, loss_dict, is_best=False)

            global_step += 1
            if global_step >= opts.max_steps:
                print('Finished training!')
                break


# ---------- Utility functions ---------- #

def validate(opts, net, orthogonal, sparse, lpips, valid_loader, device, global_step, logger=None):
    net.eval()
    agg_loss_dict = []
    for batch_idx, batch in enumerate(valid_loader):
        x, y, av_codes = batch
        with torch.no_grad():
            x, y, av_codes = x.to(device).float(), y.to(device).float(), av_codes.to(device).float()
            outputs = net.forward(x, av_codes, return_latents=True)
            loss, cur_loss_dict, id_logs = calc_loss(opts, outputs, y, orthogonal, sparse, lpips)
        agg_loss_dict.append(cur_loss_dict)

        if logger is not None:
            parse_and_log_images(opts, logger, global_step, id_logs, x, y, outputs['y_hat'],
                                 title='images/valid', subscript=f'{batch_idx:04d}')
        if global_step == 0 and batch_idx >= 4:
            net.train()
            return None

    net.train()
    loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
    return loss_dict


def calc_loss(opts, outputs, y, orthogonal, sparse, lpips):
    loss_dict = {}
    loss = 0.0
    id_logs = None

    # Resize target to match output for all pixel-wise losses
    y_resized = F.interpolate(y, size=outputs['y_hat'].shape[-2:], mode='bilinear', align_corners=False)

    if opts.l2_lambda > 0:
        loss_l2 = F.mse_loss(outputs['y_hat'], y_resized)
        loss_dict['loss_l2'] = loss_l2
        loss += loss_l2 * opts.l2_lambda

    if opts.lpips_lambda > 0:
        loss_lpips = lpips(outputs['y_hat'], y_resized)
        loss_dict['loss_lpips'] = loss_lpips
        loss += loss_lpips * opts.lpips_lambda

    if opts.orthogonal_lambda > 0:
        loss_orthogonal_AB = orthogonal(outputs['A'])
        loss_dict['loss_orthogonal'] = loss_orthogonal_AB
        loss += loss_orthogonal_AB * opts.orthogonal_lambda

    if opts.sparse_lambda > 0:
        loss_sparse = sparse(outputs['x'])
        loss_dict['loss_sparse'] = loss_sparse
        loss += loss_sparse * opts.sparse_lambda

    loss_dict['loss'] = loss
    return loss, loss_dict, id_logs


def checkpoint_me(net, opts, checkpoint_dir, best_val_loss, global_step, loss_dict, is_best):
    save_name = 'best_model.pt' if is_best else f'iteration_{global_step}.pt'
    save_dict = {
        'state_dict': net.state_dict(),
        'opts': vars(opts),
        'latent_avg': net.latent_avg
    }
    checkpoint_path = os.path.join(checkpoint_dir, save_name)
    torch.save(save_dict, checkpoint_path)
    with open(os.path.join(checkpoint_dir, 'timestamp.txt'), 'a') as f:
        if is_best:
            f.write(f'**Best**: Step - {global_step}, Loss - {best_val_loss} \n{loss_dict}\n')
        else:
            f.write(f'Step - {global_step}, \n{loss_dict}\n')


def log_metrics(logger, global_step, metrics_dict, prefix):
    for key, value in metrics_dict.items():
        logger.add_scalar(f'{prefix}/{key}', float(value), global_step)


def print_metrics(global_step, metrics_dict, prefix):
    print(f'Metrics for {prefix}, step {global_step}')
    for key, value in metrics_dict.items():
        print(f'\t{key} = {value}')


def parse_and_log_images(opts, logger, global_step, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
    im_data = []
    for i in range(display_count):
        cur_im_data = {
            'input_face': common.log_input_image(x[i], opts),
            'target_face': common.tensor2im(y[i]),
            'output_face': common.tensor2im(y_hat[i]),
        }
        if id_logs is not None:
            for key in id_logs[i]:
                cur_im_data[key] = id_logs[i][key]
        im_data.append(cur_im_data)

    # Log to TensorBoard and save as .jpg
    log_images(logger, global_step, title, im_data=im_data, subscript=subscript)


def log_images(logger, global_step, name, im_data, subscript=None, log_latest=False):
    import numpy as np
    import torch

    fig = common.vis_faces(im_data)

    # Save the figure as an image file
    step = 0 if log_latest else global_step
    path = os.path.join(logger.log_dir, name, f'{subscript}_{step:04d}.jpg') if subscript else os.path.join(logger.log_dir, name, f'{step:04d}.jpg')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)

    # Convert figure to tensor for TensorBoard
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf, _ = fig.canvas.print_to_buffer()  # returns RGBA buffer
    img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    img_arr = img_arr[:, :, :3]  # discard alpha
    img_arr = np.transpose(img_arr, (2, 0, 1))  # HWC -> CHW
    img_tensor = torch.tensor(img_arr)

    logger.add_image(name if subscript is None else f'{name}/{subscript}', img_tensor, global_step)

    plt.close(fig)


if __name__ == '__main__':
    train()
