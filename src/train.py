# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import re
import shutil
import tempfile
import warnings

import hydra
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig

from src import dnnlib
from src.training import training_loop
from src.metrics import metric_main
from src.torch_utils import training_stats
from src.torch_utils import custom_ops
from src.training.rendering import validate_frustum

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    c.run_dir = os.path.join(outdir, 'output')

    # Print options.
    print()
    if c.cfg.env.get('symlink_output', None):
        print(f'Output directory:    {c.run_dir} (symlinked to {c.cfg.env.symlink_output})')
    else:
        print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:          {c.num_gpus}')
    print(f'Batch size:              {c.batch_size} images')
    print(f'Training duration:       {c.total_kimg} kimg')
    print(f'Dataset path:            {c.training_set_kwargs.path}')
    print(f'Dataset size:            {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:      {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:          {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:         {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    if c.cfg.env.get('symlink_output', None):
        if os.path.exists(c.cfg.env.symlink_output) and not c.resume_whole_state:
            print(f'Deleting old output dir: {c.cfg.env.symlink_output} ...')
            shutil.rmtree(c.cfg.env.symlink_output)

        if not c.resume_whole_state:
            os.makedirs(c.cfg.env.symlink_output, exist_ok=False)
            os.symlink(c.cfg.env.symlink_output, c.run_dir)
            print(f'Symlinked `output` into `{c.cfg.env.symlink_output}`')
        else:
            print(f'Did not symlink `{c.cfg.env.symlink_output}` since resuming training.')
    else:
        os.makedirs(c.run_dir, exist_ok=True)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset(cfg: DictConfig):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='src.training.dataset.ImageFolderDataset', path=cfg.dataset.path, use_labels=True, max_size=None, xflip=False)
        dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset) # Be explicit about dataset size.

        if cfg.dataset.camera.dist == 'custom':
            print('Validating camera poses in the dataset...', end='')
            camera_angles = torch.from_numpy(np.array([dataset.get_camera_angles(i) for i in range(len(dataset))]))
            mean_camera_pose = camera_angles.mean(axis=0) # [3]
            assert camera_angles[:, [0]].pow(2).sum().sqrt() > 0.1, "Broken yaw angles (all zeros)."
            assert camera_angles[:, [1]].pow(2).sum().sqrt() > 0.1, "Broken pitch angles (all zeros)."
            assert camera_angles[:, [0]].min() >= -np.pi, f"Broken yaw angles (too small): {camera_angles[:, [0]].min()}"
            assert not torch.any(camera_angles[:, [0]] > np.pi), f"Number of broken yaw angles (too large): {torch.sum(camera_angles[:, [0]] > np.pi)}"
            assert camera_angles[:, [1]].min() >= 0.0, f"Broken pitch angles (too small): {camera_angles[:, [1]].min()}"
            assert not torch.any(camera_angles[:, [1]] > np.pi), f"Number of broken pitch angles (too large): {torch.sum(camera_angles[:, [1]] > np.pi)}"
            print('done!')
        else:
            mean_camera_pose = torch.tensor([cfg.dataset.camera.horizontal_mean, cfg.dataset.camera.vertical_mean, 0.0]) # [3]

        return dataset_kwargs, mean_camera_pose
    except IOError as err:
        raise ValueError(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@hydra.main(config_path="..", config_name="experiment_config.yaml")
def main(cfg: DictConfig):
    # Initialize config.
    OmegaConf.set_struct(cfg, True)

    opts = cfg.training # Training arguments.
    c = dnnlib.EasyDict(cfg=cfg) # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, cfg=cfg.model.generator, z_dim=512, w_dim=cfg.model.generator.w_dim, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_discriminator.Discriminator', cfg=cfg.model.discriminator, block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=list(cfg.model.generator.optim.betas), eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=list(cfg.model.discriminator.optim.betas), eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', cfg=cfg)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, mean_camera_pose = init_dataset(cfg)
    if opts.use_labels and not c.training_set_kwargs.use_labels:
        raise ValueError('--use_labels=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.use_labels
    c.training_set_kwargs.xflip = opts.mirror

    # Hyperparameters & settings.
    c.num_gpus = cfg.num_gpus
    c.batch_size = opts.batch_size
    c.batch_gpu = opts.batch_gpu or opts.batch_size // cfg.num_gpus
    c.G_kwargs.channel_base = int(cfg.model.generator.cbase * cfg.model.generator.fmaps)
    c.D_kwargs.channel_base = int(cfg.model.discriminator.cbase * cfg.model.discriminator.fmaps)
    c.G_kwargs.channel_max = cfg.model.generator.cmax
    c.D_kwargs.channel_max = cfg.model.discriminator.cmax
    c.G_kwargs.mapping_kwargs.num_layers = cfg.model.generator.map_depth
    c.G_kwargs.mapping_kwargs.camera_cond = cfg.model.generator.camera_cond
    c.G_kwargs.mapping_kwargs.camera_cond_drop_p = cfg.model.generator.camera_cond_drop_p
    c.G_kwargs.mapping_kwargs.camera_cond_noise_std = cfg.model.generator.camera_cond_noise_std
    c.G_kwargs.mapping_kwargs.mean_camera_pose = mean_camera_pose
    # c.D_kwargs.mapping_kwargs.mean_camera_pose = mean_camera_pose
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = cfg.model.discriminator.mbstd_group_size
    c.loss_kwargs.r1_gamma = (0.0002 * (cfg.dataset.resolution ** 2) / opts.batch_size) if opts.gamma == 'auto' else opts.gamma
    c.G_opt_kwargs.lr = cfg.model.generator.optim.lr
    c.D_opt_kwargs.lr = cfg.model.discriminator.optim.lr
    c.metrics = [] if opts.metrics is None else opts.metrics.split(',')
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    c.G_reg_interval = 4 if cfg.model.loss_kwargs.pl_weight > 0 else 0 # Enable lazy regularization for G.
    c.D_reg_interval = 16 if c.loss_kwargs.r1_gamma > 0 else None

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise ValueError('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise ValueError('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < cfg.model.discriminator.mbstd_group_size:
        raise ValueError('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise ValueError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    if cfg.model.discriminator.camera_cond:
        assert cfg.dataset.camera.dist == 'custom', f"To condition D on real camera angles, they should be available in the dataset."

    # Base configuration.
    c.ema_kimg = c.batch_size * cfg.model.generator.ema_multiplier
    if cfg.model.name == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    elif cfg.model.name == 'epigraf':
        c.G_kwargs.class_name = 'training.networks_epigraf.Generator'
        if cfg.model.generator.backbone == 'stylegan2':
            c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        if cfg.model.generator.backbone == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.

        print('Validating that the vieweing frustum is inside the cube...', end='')
        assert validate_frustum(
            fov=cfg.dataset.camera.fov,
            near=cfg.dataset.camera.ray_start,
            far=cfg.dataset.camera.ray_end,
            radius=cfg.dataset.camera.radius,
            scale=cfg.dataset.get('cube_scale', 1.0),
            verbose=False,
        ), f"Please, increase the scale: {cfg.model.generator.tri_plane.scale}"
        print('Done!')
    elif cfg.model.name == 'inr-gan':
        del c.G_kwargs.channel_base
        del c.G_kwargs.channel_max
        c.G_kwargs.class_name = 'training.networks_inr_gan.Generator'
    elif cfg.model.name in ['stylegan3-t', 'stylegan3-r']:
        c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if cfg.model.name == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
    else:
        raise NotImplementedError(f'Unknown model: {cfg.model.name}')

    # Augmentation.
    if opts.augment.mode != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **opts.augment.probs)
        if opts.augment.mode == 'ada':
            c.ada_target = opts.target
        if opts.augment.mode == 'fixed':
            c.augment_p = opts.p

    if cfg.training.patch.enabled:
        if cfg.training.patch.distribution in ('uniform', 'discrete_uniform', 'beta'):
            assert cfg.training.patch.min_scale_trg * cfg.dataset.resolution >= cfg.training.patch.resolution, \
                f"It does not make any sense to have so small patch size of {cfg.training.patch.min_scale_trg} " \
                f"at resolution {cfg.training.patch.resolution} when the dataset resolution is just {cfg.dataset.resolution}"

    # Resume.
    c.resume_whole_state = False
    if opts.resume == 'latest':
        ckpt_regex = re.compile("^network-snapshot-\d{6}.pkl$")
        run_dir = os.path.join(cfg.experiment_dir, 'output')
        ckpts = sorted([f for f in os.listdir(run_dir) if ckpt_regex.match(f)]) if os.path.isdir(run_dir) else []

        if len(ckpts) > 0:
            c.resume_pkl = os.path.join(run_dir, ckpts[-1])
            c.resume_whole_state = True
            print(f'Will resume training from {ckpts[-1]}')
        else:
            warnings.warn("Was requested to continue training, but couldn't find any checkpoints. Please remove `training.resume=latest` argument.")
    elif opts.resume is not None:
        c.resume_pkl = opts.resume
        if opts.resume_only_G:
            c.ada_kimg = 100 # Make ADA react faster at the beginning.
            c.ema_rampup = None # Disable EMA rampup.
            c.cfg.model.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
        else:
            print('Will load whole state from {c.resume_pkl}')
            c.resume_whole_state = True

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Launch.
    launch_training(c=c, outdir=cfg.experiment_dir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
