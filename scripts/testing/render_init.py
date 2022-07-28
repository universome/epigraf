"""
This script computes imgs/sec for a generator in the eval mode
for different batch sizes
"""
import sys; sys.path.extend(['..', '.', 'src'])

import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from torchvision import utils
import torchvision.transforms.functional as TVF

from src import dnnlib
from src.infra.utils import recursive_instantiate


def instantiate_G(cfg: DictConfig, use_grad: bool=False) -> nn.Module:
    G_kwargs = dnnlib.EasyDict(class_name=None, cfg=cfg.model.generator, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    G_kwargs.mapping_kwargs.num_layers = cfg.model.generator.map_depth
    G_kwargs.mapping_kwargs.camera_cond = cfg.training.get('camera_cond', False)
    G_kwargs.mapping_kwargs.camera_cond_drop_p = cfg.training.get('camera_cond_drop_p', 0.0)
    G_kwargs.mapping_kwargs.camera_cond_noise_std = cfg.training.get('camera_cond_noise_std', 0.0)

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

    G = dnnlib.util.construct_class_by_name(**G_kwargs).eval().requires_grad_(use_grad)

    return G


@hydra.main(config_path="../../../configs", config_name="config.yaml")
def profile(cfg: DictConfig):
    recursive_instantiate(cfg)
    device = 'cuda'
    batch_size = cfg.training.batch_size
    all_imgs = []
    save_path = '/home/skoroki/rnf/fakes_init.jpg'

    for i in range(4):
        G = instantiate_G(cfg, use_grad=False).to(device)

        with torch.set_grad_enabled(mode=False):
            z = torch.randn(batch_size, G.z_dim, device=device)
            c = torch.zeros(batch_size, G.c_dim, device=device)
            camera_angles = torch.zeros(batch_size, 3, device=device)
            img = G(z, c, camera_angles, ignore_bg=cfg.get('ignore_bg', False), bg_only=cfg.get('bg_only', False))
            img = img.clamp(-1, 1).cpu() * 0.5 + 0.5 # [b, c, h, w]
            all_imgs.extend(img)
    img = utils.make_grid(torch.stack(all_imgs), nrow=int(np.sqrt(len(all_imgs))))
    TVF.to_pil_image(img).save(save_path, q=95)

    print(f'Saved into {save_path}')


if __name__ == '__main__':
    profile()
