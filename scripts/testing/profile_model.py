"""
This script computes imgs/sec for a generator for different batch sizes
"""
import sys; sys.path.extend(['..', '.', 'src'])
import time

import numpy as np
import torch
import torch.nn as nn
import hydra
from hydra.experimental import initialize
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch.autograd.profiler as profiler

from src import dnnlib
from src.infra.utils import recursive_instantiate


DEVICE = 'cuda'


def instantiate_G(cfg: DictConfig, use_grad: bool=False, train_mode: bool=False) -> nn.Module:
    G_kwargs = dnnlib.EasyDict(class_name=None, cfg=cfg.model.generator, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    G_kwargs.channel_base = cfg.model.generator.cbase
    G_kwargs.channel_max = cfg.model.generator.cmax
    G_kwargs.mapping_kwargs.num_layers = cfg.model.generator.map_depth

    if cfg.model.name == 'stylegan2':
        G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    elif cfg.model.name == 'epigraf':
        G_kwargs.class_name = 'training.networks_epigraf.Generator'
        G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    else:
        G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        G_kwargs.magnitude_ema_beta = 0.5 ** (cfg.training.batch_size / (20 * 1e3))
        if cfg.model.name == 'stylegan3-r':
            G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            G_kwargs.channel_base *= 2 # Double the number of feature maps.
            G_kwargs.channel_max *= 2
            G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
    G_kwargs.c_dim = 0
    G_kwargs.img_resolution = cfg.get('resolution', 256)
    G_kwargs.img_channels = 3

    if cfg.training.fp32:
        G_kwargs.num_fp16_res = 0
        G_kwargs.conv_clamp = None

    G = dnnlib.util.construct_class_by_name(**G_kwargs).train(train_mode).requires_grad_(use_grad).to(DEVICE)

    return G


def profile_for_batch_size(G: nn.Module, cfg: DictConfig, batch_size: int, run_backward: bool=False, profiling_row_limit: int=10):
    z = torch.randn(batch_size, G.z_dim, device=DEVICE)
    c = torch.zeros(batch_size, G.c_dim, device=DEVICE)
    camera_angles = torch.zeros(batch_size, 3, device=DEVICE)
    times = []

    for i in tqdm(range(cfg.get('num_warmup_iters', 5)), desc='Warming up'):
        torch.cuda.synchronize()
        fake_img = G(z, c=c, camera_angles=camera_angles).contiguous()
        y = fake_img[0, 0, 0, 0].item() # sync
        torch.cuda.synchronize()

    time.sleep(1)

    torch.cuda.reset_peak_memory_stats()

    with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        for i in tqdm(range(cfg.get('num_profile_iters', 25)), desc='Profiling'):
            torch.cuda.synchronize()
            start_time = time.time()
            with profiler.record_function("forward"):
                fake_img = G(z, c=c, camera_angles=camera_angles).contiguous()
                y = fake_img[0, 0, 0, 0].item() # sync
                torch.cuda.synchronize()
                if not run_backward:
                    times.append(time.time() - start_time)

            if run_backward:
                torch.cuda.synchronize()
                with profiler.record_function("forward"):
                    fake_img.sum().backward()
                    torch.cuda.synchronize()
                    times.append(time.time() - start_time)

    torch.cuda.empty_cache()
    num_imgs_processed = len(times) * batch_size
    total_time_spent = np.sum(times)
    bandwidth = num_imgs_processed / total_time_spent
    summary = prof.key_averages().table(sort_by="cpu_time_total", row_limit=profiling_row_limit)

    print(f'[Batch size: {batch_size}] Mean: {np.mean(times):.05f}s/it. Std: {np.std(times):.05f}s')
    print(f'[Batch size: {batch_size}] Imgs/sec: {bandwidth:.03f}')
    print(f'[Batch size: {batch_size}] Max mem: {torch.cuda.max_memory_allocated(DEVICE) / 2**30:<6.2f} gb')

    return bandwidth, summary


@hydra.main(config_path="../../configs", config_name="config.yaml")
def profile(cfg: DictConfig):
    recursive_instantiate(cfg)
    use_grad = cfg.get('use_grad', False)
    run_backward = cfg.get('run_backward', False)
    train_mode = cfg.get('train_mode', False)
    G = instantiate_G(cfg, use_grad=use_grad, train_mode=train_mode)
    batch_sizes = [cfg.training.batch_size]
    bandwidths = []
    summaries = []
    print(f'Number of parameters: {sum(p.numel() for p in G.parameters())}')

    with torch.set_grad_enabled(mode=use_grad):
        for batch_size in batch_sizes:
            bandwidth, summary = profile_for_batch_size(G, cfg, batch_size, run_backward=run_backward, profiling_row_limit=cfg.get('profiling_row_limit', 10))
            bandwidths.append(bandwidth)
            summaries.append(summary)

    best_batch_size_idx = int(np.argmax(bandwidths))
    print(f'------------ Best batch size is {batch_sizes[best_batch_size_idx]} ------------')
    print(summaries[best_batch_size_idx])


if __name__ == '__main__':
    profile()
