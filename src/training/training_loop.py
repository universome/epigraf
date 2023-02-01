# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import src.legacy
from src import dnnlib
from src.torch_utils import misc
from src.torch_utils import training_stats
from src.torch_utils.ops import conv2d_gradfix
from src.torch_utils.ops import grid_sample_gradfix
from src.metrics import metric_main
from src.training.rendering import sample_camera_angles
from src.training.inference_utils import (
    setup_snapshot_image_grid,
    save_image_grid,
    generate_videos,
    save_videos,
)

#----------------------------------------------------------------------------

def training_loop(
    cfg: DictConfig         = {},       # Main config
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_whole_state      = False,    # Should we resume the whole state?
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    assert cfg.training.main_metric == '__pick_first__', f'Cant select a particular metric: {cfg.training.main_metric}'

    if cfg.run_profiling:
        # Initialize the profiler
        print(f'Initializing the profiler (on rank: {rank})')
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=3, active=3, repeat=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(run_dir, 'profiling_logs')),
            record_shapes=True,
            with_stack=True)
        profiler.start()
    else:
        profiler = None

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=cfg.dataset.c_dim, img_channels=training_set.num_channels)
    D_img_resolution = loss_kwargs.cfg.training.patch.resolution if loss_kwargs.cfg.training.patch.enabled else training_set.resolution
    G = dnnlib.util.construct_class_by_name(img_resolution=training_set.resolution,  **G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(img_resolution=D_img_resolution, **D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if not resume_pkl is None:
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = src.legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Initialize logging iterations
    cur_nimg = resume_data['stats']['cur_nimg'] if resume_whole_state else 0
    cur_tick = resume_data['stats']['cur_tick'] if resume_whole_state else 0
    batch_idx = resume_data['stats']['batch_idx'] if resume_whole_state else 0
    best_metric_value = resume_data['stats']['best_metric_value'] if resume_whole_state else float('inf')  # Tracking the best metric value
    best_metric_tick = resume_data['stats']['best_metric_tick'] if resume_whole_state else 0               # Tracking at which kimg the best metric was encountered
    best_metric_nimg = resume_data['stats']['best_metric_nimg'] if resume_whole_state else 0               # Tracking at which nimg the best metric was encountered

    # Print network summary tables.
    if rank == 0:
        with torch.no_grad():
            G.eval(); D.eval()
            z = torch.empty([cfg.training.test_batch_gpu, G.z_dim], device=device)
            c = torch.empty([cfg.training.test_batch_gpu, G.c_dim], device=device)
            camera_angles = torch.empty([cfg.training.test_batch_gpu, 3], device=device) # [batch_size, 3]
            img = misc.print_module_summary(G, [z[[0]], c[[0]]], module_kwargs={'camera_angles': camera_angles[[0]]}) # [1, c, h, w]
            img = img.repeat(cfg.training.test_batch_gpu, 1, 1, 1) # [batch_size, c, h, w]
            if loss_kwargs.cfg.training.patch.enabled:
                img = img[:, :, :loss_kwargs.cfg.training.patch.resolution, :loss_kwargs.cfg.training.patch.resolution] # [batch_size, c, patch_h, patch_w]
            misc.print_module_summary(D, [img, c], module_kwargs={
                'patch_params': {
                    'scales': torch.zeros(cfg.training.test_batch_gpu, 2, device=device),
                    'offsets': torch.zeros(cfg.training.test_batch_gpu, 2, device=device),
                },
                'camera_angles': torch.zeros(cfg.training.test_batch_gpu, 3, device=device),
            })
            G.train(); D.train()

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')
        if resume_whole_state:
            misc.copy_params_and_buffers(resume_data['augment_pipe'], augment_pipe, require_all=False)
    else:
        augment_pipe = None
        ada_stats = None

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []

    # TODO: it makes sense to separate G and D phases construction?
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval in [None, 0]:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'all', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            if name == 'G':
                assert cfg.model.loss_kwargs.pl_weight > 0
                phases += [dnnlib.EasyDict(name='Greg_pl', module=module, opt=opt, interval=reg_interval)]
            else:
                phases += [dnnlib.EasyDict(name='Dreg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Ok, we need to extract G_opt and D_opt back from phases since we want to save them...
    # We load them here since the checkpoints are ready
    G_opt = next(p.opt for p in phases if p.name in {'Gall', 'Gmain'})
    D_opt = next(p.opt for p in phases if p.name in {'Dall', 'Dmain'})
    if resume_whole_state:
        G_opt.load_state_dict(resume_data['G_opt'].state_dict())
        D_opt.load_state_dict(resume_data['D_opt'].state_dict())

    # Export sample images.
    if rank == 0:
        if not resume_whole_state:
            print('Exporting sample images/videos...')
            vis = dnnlib.EasyDict()
            vis.grid_size, images, vis.labels, vis.camera_angles = setup_snapshot_image_grid(training_set=training_set, cfg=cfg)
            save_image_grid(images, os.path.join(run_dir, 'reals.jpg'), drange=[0,255], grid_size=vis.grid_size)
            vis.grid_z = torch.randn([vis.labels.shape[0], G.z_dim], device=device).split(cfg.training.test_batch_gpu) # (num_batches, [batch_size, z_dim])
            vis.grid_c = torch.from_numpy(vis.labels).to(device).split(cfg.training.test_batch_gpu) # (num_batches, [batch_size, c_dim])
            vis.grid_camera_angles = torch.from_numpy(vis.camera_angles).to(device).split(cfg.training.test_batch_gpu) # (num_batches, [batch_size, 3])
            save_filename = 'fakes_init.jpg'
        else:
            vis = dnnlib.EasyDict(**resume_data['vis'])
            for k in vis:
                if isinstance(vis[k], torch.Tensor):
                    vis[k] = vis[k].to(device)
            save_filename = f'fakes_resume_{cur_nimg:06d}.jpg'

        with torch.no_grad():
            images = torch.cat([G_ema(z=z, c=c, camera_angles=a, noise_mode='const').cpu() for z, c, a in zip(vis.grid_z, vis.grid_c, vis.grid_camera_angles)]).numpy()
            videos = generate_videos(G_ema, torch.stack(vis.grid_z).view(-1, G.z_dim), torch.stack(vis.grid_c).view(vis.labels.shape[0], G.c_dim)) # [num_videos, num_frames, c, h, w]
        save_image_grid(images, os.path.join(run_dir, save_filename), drange=[-1,1], grid_size=vis.grid_size)
        save_videos(videos, os.path.join(run_dir, save_filename.replace('.jpg', '.mp4')))
    else:
        vis = dnnlib.EasyDict()

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)

            if not resume_whole_state:
                config_yaml = OmegaConf.to_yaml(cfg)
                stats_tfevents.add_text(f'config', text_to_markdown(config_yaml), global_step=0, walltime=time.time())
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()

    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time

    if progress_fn is not None:
        progress_fn(0, total_kimg)

    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            batch = next(training_set_iterator)
            phase_real_img, phase_real_c, phase_real_camera_angles = batch['image'], batch['label'], batch['camera_angles']
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            phase_real_camera_angles = phase_real_camera_angles.to(device).split(batch_gpu) # (batch_size // batch_gpu, [batch_gpu, 3])
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            gen_cond_sample_idx = [np.random.randint(len(training_set)) for _ in range(len(phases) * batch_size)] # [num_phases * batch_size]
            all_gen_c = [training_set.get_label(i) for i in gen_cond_sample_idx] # [num_phases * batch_size]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device) # [num_phases * batch_size]
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)] # [num_phases, batch_size // batch_gpu, batch_gpu]
            if cfg.dataset.camera.dist == 'custom':
                all_gen_camera_angles = [training_set.get_camera_angles(i) for i in gen_cond_sample_idx] # [N, 3]
                all_gen_camera_angles = torch.from_numpy(np.stack(all_gen_camera_angles)).pin_memory().to(device) # [N, 3]
            else:
                all_gen_camera_angles = sample_camera_angles(cfg=cfg.dataset.camera, batch_size=len(gen_cond_sample_idx), device=device) # [N, 3]

            # Preparing GPC data (camera conditioning for G)
            # Shift the values in X% of random places by 1 to spoof the generator
            # Does not work for some reason :|
            # TODO: maybe spoofing is bad?
            all_gen_camera_angles_cond = all_gen_camera_angles.clone() # [N, 3]
            camera_spoof_idx = torch.rand(all_gen_camera_angles_cond.shape[0]) < loss.gpc_spoof_p # [N]
            all_gen_camera_angles_cond[camera_spoof_idx] = all_gen_camera_angles_cond[camera_spoof_idx].roll(shifts=1, dims=0) # [~N/2, 3]

            all_gen_camera_angles = [phase_cs.split(batch_gpu) for phase_cs in all_gen_camera_angles.split(batch_size)]
            all_gen_camera_angles_cond = [phase_cs.split(batch_gpu) for phase_cs in all_gen_camera_angles_cond.split(batch_size)]

        # Execute training phases.
        all_data = zip(phases, all_gen_z, all_gen_c, all_gen_camera_angles, all_gen_camera_angles_cond)
        for (phase, phase_gen_z, phase_gen_c, phase_gen_camera_angles, phase_gen_camera_angles_cond) in all_data:
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            phase_data = zip(
                phase_real_img,
                phase_real_c,
                phase_real_camera_angles,
                phase_gen_z,
                phase_gen_c,
                phase_gen_camera_angles,
                phase_gen_camera_angles_cond,
            )
            for real_img, real_c, real_camera_angles, gen_z, gen_c, gen_camera_angles, gen_camera_angles_cond in phase_data:
                loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    real_c=real_c,
                    real_camera_angles=real_camera_angles,
                    gen_z=gen_z,
                    gen_c=gen_c,
                    gen_camera_angles=gen_camera_angles,
                    gen_camera_angles_cond=gen_camera_angles_cond,
                    gain=phase.interval,
                    cur_nimg=cur_nimg
                )
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                if phase.name in ['Gmain', 'Gall'] and not G.cfg.optim.get('grad_clip', None) is None:
                    torch.nn.utils.clip_grad_norm_(params, G.cfg.optim.grad_clip)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        G.progressive_update(cur_nimg / 1000)
        loss.progressive_update(cur_nimg / 1000)
        if not profiler is None:
            profiler.step()

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (cfg.training.image_snap is not None) and (done or cur_tick % cfg.training.image_snap == 0):
            with torch.no_grad():
                images = torch.cat([G_ema(z=z, c=c, camera_angles=a, noise_mode='const').cpu() for z, c, a in zip(vis.grid_z, vis.grid_c, vis.grid_camera_angles)]).numpy()
                videos = generate_videos(G_ema, torch.stack(vis.grid_z).view(-1, G.z_dim), torch.stack(vis.grid_c).view(vis.labels.shape[0], G.c_dim)) # [num_videos, num_frames, c, h, w]
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.jpg'), drange=[-1,1], grid_size=vis.grid_size)
            save_videos(videos, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.mp4'))

        # Save network snapshot.
        snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
        snapshot_data = None
        snapshot_modules = [
            ('G', G),
            ('D', D),
            ('G_ema', G_ema),
            ('augment_pipe', augment_pipe),
            ('G_opt', G_opt),
            ('D_opt', D_opt),
            ('vis', {k: (v.to('cpu') if isinstance(v, torch.Tensor) else v) for k, v in vis.items()}),
            ('stats', {
                'cur_nimg': cur_nimg,
                'cur_tick': cur_tick,
                'batch_idx': batch_idx,
                'best_metric_value': best_metric_value,
                'best_metric_tick': best_metric_tick,
                'best_metric_nimg': best_metric_nimg,
            }),
            ('training_set_kwargs', dict(training_set_kwargs)),
        ]
        # Checking the ddp consistency
        # TODO: somehow, if we'll put it inside the snapshot saving code, it fails with DDP
        if cur_tick % cfg.training.snap == 0:
            DDP_CONSISTENCY_IGNORE_REGEX = r'.*\.[^.]+_(avg|ema)'
            for name, module in snapshot_modules:
                if module is not None:
                    if isinstance(module, torch.nn.Module):
                        if num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=DDP_CONSISTENCY_IGNORE_REGEX)
                            for param in misc.params_and_buffers(module):
                                torch.distributed.broadcast(param, src=0)
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    else:
                        module = copy.deepcopy(module)
        # Evaluate metrics.
        curr_main_metric_val = float('inf')
        if len(metrics) > 0 and cur_tick % cfg.training.val_freq == 0:
            if rank == 0:
                print(f'Evaluating metrics for {run_dir} ...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=G_ema, batch_gen=cfg.training.test_batch_gpu,
                    dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if cfg.training.main_metric == '__pick_first__' and metric == metrics[0]:
                    curr_main_metric_val = result_dict['results'][metric]
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        # Save the checkpoint
        if done or cur_tick % cfg.training.snap == 0 or curr_main_metric_val <= best_metric_value:
            if rank == 0:
                snapshot_reason = "done" if done else ("tick" if cur_tick % cfg.training.snap == 0 else "best metric")
                print(f'Saving the snapshot... (reason: {snapshot_reason}) ', end='')
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            DDP_CONSISTENCY_IGNORE_REGEX = r'.*\.[^.]+_(avg|ema)'
            for name, module in snapshot_modules:
                snapshot_data[name] = module
                del module # conserve memory
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)
                print('Saved!')
        if rank == 0 and curr_main_metric_val <= best_metric_value:
            prev_best_ckpt_path = os.path.join(run_dir, f'network-snapshot-{best_metric_nimg//1000:06d}.pkl')
            if best_metric_tick % cfg.training.snap == 0 or done:
                # Do not delete the snapshot since we would save it anyway
                pass
            elif os.path.isfile(prev_best_ckpt_path):
                # Deleting the previous best
                os.remove(prev_best_ckpt_path)
            # Updating the best ckpt statistics
            best_metric_tick = cur_tick
            best_metric_nimg = cur_nimg
            best_metric_value = curr_main_metric_val
        del snapshot_data # conserve memory
        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=timestamp)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=timestamp)
            if hasattr(G.synthesis, 'nerf_noise_std'):
                stats_tfevents.add_scalar('Progress/nerf_noise_std', G.synthesis.nerf_noise_std, global_step=global_step, walltime=timestamp)
            if hasattr(G.synthesis, 'nerf_sp_beta'):
                stats_tfevents.add_scalar('Progress/nerf_sp_beta', G.synthesis.nerf_sp_beta, global_step=global_step, walltime=timestamp)
            if hasattr(loss, 'patch_cfg') and 'min_scale' in loss.patch_cfg:
                stats_tfevents.add_scalar('Progress/min_scale', loss.patch_cfg['min_scale'], global_step=global_step, walltime=timestamp)
            if hasattr(loss, 'patch_cfg') and 'beta' in loss.patch_cfg:
                stats_tfevents.add_scalar('Progress/beta', loss.patch_cfg['beta'], global_step=global_step, walltime=timestamp)
            stats_tfevents.add_scalar('Progress/gpc_spoof_p', loss.gpc_spoof_p, global_step=global_step, walltime=timestamp)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    if not profiler is None:
        profiler.stop()

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------

def text_to_markdown(text: str) -> str:
    """
    Converts an arbitrarily text into a text that would be well-displayed in TensorBoard.
    TensorBoard uses markdown to render the text that's why it strips spaces and line breaks.
    This function fixes that.
    """
    text = text.replace(' ', '&nbsp;&nbsp;') # Because markdown does not support text indentation normally...
    text = text.replace('\n', '  \n') # Because tensorboard uses markdown

    return text

#----------------------------------------------------------------------------
