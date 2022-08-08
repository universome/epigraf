"""
Generates a multi-view video of a sample from pi-GAN
Adapted from Adapted from https://github.com/marcoamonteiro/pi-GAN
"""

import os
from typing import List

import hydra
import torch
import numpy as np
from omegaconf import DictConfig
import torchvision as tv
from torchvision.utils import make_grid
import torchvision.transforms.functional as TVF
from tqdm import tqdm

from src.training.inference_utils import generate_trajectory, generate, generate_camera_angles
from scripts.utils import load_generator, set_seed, maybe_makedirs, lanczos_resize_tensors

torch.set_grad_enabled(False)

#----------------------------------------------------------------------------

@hydra.main(config_path="../configs/scripts", config_name="inference.yaml")
def generate_vis(cfg: DictConfig):
    device = torch.device('cuda')
    save_dir = os.path.join(cfg.output_dir, cfg.vis.name)
    set_seed(42) # To fix non-z randomization

    G = load_generator(cfg.ckpt, verbose=cfg.verbose)[0].to(device).eval()
    G.synthesis.img_resolution = G.synthesis.test_resolution = cfg.img_resolution
    G.cfg.num_ray_steps = G.cfg.num_ray_steps * cfg.ray_step_multiplier
    G.cfg.bg_model.num_steps = G.cfg.bg_model.num_steps * cfg.ray_step_multiplier
    G.cfg.dataset.white_back = True if cfg.force_whiteback else G.cfg.dataset.white_back
    G.nerf_noise_std = 0
    maybe_makedirs(save_dir)
    assert (not cfg.seeds is None) or (not cfg.num_seeds is None), "You must specify either `num_seeds` or `seeds`"
    seeds = cfg.seeds if cfg.num_seeds is None else np.arange(cfg.num_seeds)

    if cfg.vis.name == 'front_grid':
        ws = sample_ws_from_seeds(G, seeds, cfg, device) # [num_grids, num_ws, w_dim]
        trajectory = generate_camera_angles(cfg.camera, default_fov=None)[0][:, :2] # [num_grids, 2]
        images = generate_trajectory(cfg, G, ws, trajectory) # [num_traj_steps, num_samples, c, h, w]
        images = images.permute(1, 0, 2, 3, 4) # [num_samples, num_traj_steps, c, h, w]

        for seed, grid in tqdm(list(zip(seeds, images)), desc='Saving'):
            pad_size, num_angles, h, w = 2, grid.shape[0], grid.shape[2], grid.shape[3]
            h_small, w_small = h // (num_angles - 1), w // (num_angles - 1)
            main_img = grid[[0]] # [1, c, h, w]
            small_imgs = grid[1:] # [num_angles - 1, c, h, w]
            small_imgs = lanczos_resize_tensors(small_imgs, size=(h_small, w_small)) # [num_angles - 1, c, h_small, w_small]
            main_img = lanczos_resize_tensors(main_img, size=(h + pad_size, w + pad_size))
            subgrid = make_grid(small_imgs, nrow=len(trajectory) - 1) # [c, grid_h, grid_w]
            grid = torch.cat([
                TVF.pad(main_img, padding=(pad_size, pad_size, pad_size, 0)).squeeze(0),
                subgrid
            ], dim=1) # [c, h + 2, grid_h]
            TVF.to_pil_image(grid).save(os.path.join(save_dir, f'seed-{seed:04d}.jpg'), q=95)
    elif cfg.vis.name == 'density':
        ws = sample_ws_from_seeds(G, seeds, cfg, device) # [num_objects, num_ws, w_dim]
        density = generate_density(cfg, G, ws) # [num_objects, resolution, resolution, resolution]
        trajectory = torch.tensor([cfg.yaws[0], np.pi * 3 / 8]).unsqueeze(0).numpy() # [1, 2]
        images = generate_trajectory(cfg, G, ws, trajectory).squeeze(0) # [num_samples, c, h, w]
        for seed, density_field, img in tqdm(list(zip(seeds, density, images)), desc='Saving'):
            np.save(os.path.join(save_dir, f'seed-{seed:04d}'), density_field)
            TVF.to_pil_image(img).save(os.path.join(save_dir, f'seed-{seed:04d}.jpg'), q=95)
    elif cfg.vis.name == 'interp':
        trajectory = torch.tensor([0.0, np.pi * 3 / 8]).unsqueeze(0).numpy() # [1, 3]
        ws = sample_ws_from_seeds(G, seeds, cfg, device, cfg.num_interp_steps) # [num_interp_steps, num_grids, num_ws, w_dim]
        ws = ws.reshape(cfg.num_interp_steps * ws.shape[1], *ws.shape[2:]) # [num_interp_steps * num_grids, num_ws, w_dim]
        images = generate_trajectory(cfg, G, ws, trajectory) # [num_traj_steps, num_interp_steps * num_grids, c, h, w]
        images = images.reshape(cfg.num_interp_steps, len(seeds) // 2, *images.shape[2:]) # [num_interp_steps, num_grids, c, h, w]
        images = images.permute(1, 0, 2, 3, 4) # [num_grids, num_interp_steps, c, h, w]

        for seed_from, seed_to, grid in tqdm(list(zip(seeds[0::2], seeds[1::2], images)), desc='Saving'):
            grid = make_grid(grid, nrow=cfg.num_interp_steps) # [c, grid_h, grid_w]
            TVF.to_pil_image(grid).save(os.path.join(save_dir, f'seeds-{seed_from:04d}-{seed_to:04d}.jpg'), q=95)
    elif cfg.vis.name == 'interp_video':
        """Does 'live' interpolation, i.e. an object rotates and transforms into another one simuletaneously"""
        num_videos = len(seeds) // 2 # [1]
        angles = generate_camera_angles(cfg.camera, default_fov=None)[0] # [num_videos, 3]
        ws = sample_ws_from_seeds(G, seeds, cfg, device, num_interp_steps=cfg.camera.num_frames) # [num_frames, num_videos, num_ws, w_dim]
        ws = ws.reshape(cfg.camera.num_frames * num_videos, *ws.shape[2:]) # [num_frames * num_videos, num_ws, w_dim]
        angles = torch.from_numpy(angles).float().to(device).repeat_interleave(num_videos, dim=0) # [num_frames * num_videos, 3]
        images = generate(cfg, G, ws, angles=angles) # [num_frames * num_videos, c, h, w]
        images = images.reshape(cfg.camera.num_frames, num_videos, *images.shape[1:]) # [num_frames, num_videos, c, h, w]
        images = images.permute(1, 0, 2, 3, 4) # [num_videos, num_frames, c, h, w]
        for seed_from, seed_to, video in tqdm(list(zip(seeds[0::2], seeds[1::2], images)), desc='Saving'):
            save_path = os.path.join(save_dir, f'seeds-{seed_from:04d}-{seed_to:04d}.mp4')
            video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
            tv.io.write_video(save_path, video, fps=cfg.vis.fps, video_codec='h264', options={'crf': '10'})
    elif cfg.vis.name == 'interp_video_grid':
        """Renders the interpolation as a grid, i.e. renders each interpolation step separately and fully"""
        num_grids = len(seeds) // 2 # [1]
        trajectory = generate_camera_angles(cfg.camera, default_fov=None)[0][:, :2] # [num_grids, 2]
        ws = sample_ws_from_seeds(G, seeds, cfg, device, num_interp_steps=cfg.vis.num_interp_steps) # [num_interp_steps, num_grids, num_ws, w_dim]
        ws = ws.reshape(cfg.vis.num_interp_steps * ws.shape[1], *ws.shape[2:]) # [num_interp_steps * num_grids, num_ws, w_dim]
        images = generate_trajectory(cfg, G, ws, trajectory) # [num_frames, num_interp_steps * num_grids, c, h, w]
        images = images.reshape(cfg.camera.num_frames, cfg.vis.num_interp_steps, num_grids, *images.shape[2:]) # [num_frames, num_interp_steps, num_grids, c, h, w]
        images = images.permute(2, 0, 1, 3, 4, 5) # [num_grids, num_frames, num_interp_steps, c, h, w]
        for seed_from, seed_to, grid in tqdm(list(zip(seeds[0::2], seeds[1::2], images)), desc='Saving'):
            frame_grids = torch.stack([make_grid(g, nrow=cfg.vis.num_interp_steps) for g in grid]) # [num_frames, num_interp_steps, c, h, w]
            save_path = os.path.join(save_dir, f'seeds-{seed_from:04d}-{seed_to:04d}.mp4')
            video = (frame_grids * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
            tv.io.write_video(save_path, video, fps=cfg.vis.fps, video_codec='h264', options={'crf': '10'})
    elif cfg.vis.name == 'video':
        angles, fovs = generate_camera_angles(cfg.camera, default_fov=G.cfg.dataset.camera.fov) # [num_frames, 3], [num_frames]
        ws = sample_ws_from_seeds(G, seeds, cfg, device, num_interp_steps=0) # [num_videos, num_ws, w_dim]
        images = generate_trajectory(cfg, G, ws, angles[:, :2], fovs=fovs, **cfg.synthesis_kwargs) # [num_frames, num_videos, c, h, w]
        images = images.permute(1, 0, 2, 3, 4) # [num_videos, num_frames, c, h, w]
        for seed, video_frames in tqdm(list(zip(seeds, images)), desc='Saving'):
            save_path = os.path.join(save_dir, f'seed-{seed:04d}.mp4')
            video = (video_frames * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
            tv.io.write_video(save_path, video, fps=cfg.vis.fps, video_codec='h264', options={'crf': '10'})
    elif cfg.vis.name == 'video_grid':
        angles, fovs = generate_camera_angles(cfg.camera, default_fov=G.cfg.dataset.camera.fov) # [num_frames, 3], [num_frames]
        ws = sample_ws_from_seeds(G, seeds, cfg, device, num_interp_steps=0) # [num_videos, num_ws, w_dim]
        images = generate_trajectory(cfg, G, ws, angles[:, :2], fovs=fovs, **cfg.synthesis_kwargs) # [num_frames, num_videos, c, h, w]
        for grid_idx in tqdm(list(range(0, (len(seeds) + cfg.vis.num_videos_per_grid - 1) // cfg.vis.num_videos_per_grid)), desc='Saving'):
            grid_slice = slice(grid_idx * cfg.vis.num_videos_per_grid, (grid_idx + 1) * cfg.vis.num_videos_per_grid, 1)
            grid_videos = images[:, grid_slice] # [t, num_videos_per_grid, c, h, w]
            grid_seeds = seeds[grid_slice] # [num_videos_per_grid]
            save_path = os.path.join(save_dir, f'seeds-{grid_seeds[0]:04d}-to-{grid_seeds[-1]:04d}.mp4')
            video_frames = torch.stack([make_grid(g, nrow=cfg.vis.nrow) for g in grid_videos]) # [t, c, gh, gw]
            video = (video_frames * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
            tv.io.write_video(save_path, video, fps=cfg.vis.fps, video_codec='h264', options={'crf': '10'})
    elif cfg.vis.name == 'interp_density':
        ws = sample_ws_from_seeds(G, seeds, cfg, device, cfg.num_interp_steps) # [num_interp_steps, num_videos, num_ws, w_dim]
        ws = ws.reshape(cfg.num_interp_steps * ws.shape[1], *ws.shape[2:]) # [num_interp_steps * num_videos, num_ws, w_dim]
        density = generate_density(cfg, G, ws) # [num_interp_steps * num_videos, resolution, resolution, resolution]
        density = density.reshape(cfg.num_interp_steps, len(seeds) // 2, *density.shape[1:]).transpose(1, 0, 2, 3, 4) # [num_videos, num_interp_steps, c, h, w]

        for seed_from, seed_to, density_field in tqdm(list(zip(seeds[0::2], seeds[1::2], density)), desc='Saving'):
            np.save(os.path.join(save_dir, f'seeds-{seed_from:04d}-{seed_to:04d}'), density_field)
    elif cfg.vis.name == 'bg_nobg':
        trajectory = torch.tensor([0.0, np.pi / 2]).unsqueeze(0).numpy() # [1, 2]
        ws = sample_ws_from_seeds(G, seeds, cfg, device) # [num_videos, num_ws, w_dim]
        images_bg = generate_trajectory(cfg, G, ws, trajectory, ignore_bg=False).squeeze(0) # [num_traj_steps, num_videos, c, h, w]
        G.cfg.dataset.white_back = True
        images_nobg = generate_trajectory(cfg, G, ws, trajectory, ignore_bg=True).squeeze(0) # [num_traj_steps, num_videos, c, h, w]

        for seed, img_bg, img_nobg in tqdm(list(zip(seeds, images_bg, images_nobg)), desc='Saving'):
            grid = make_grid(torch.stack([img_bg, img_nobg]))
            TVF.to_pil_image(grid).save(os.path.join(save_dir, f'seed-{seed:04d}.jpg'), q=95)
    elif cfg.vis.name == 'minigrid':
        ws = sample_ws_from_seeds(G, seeds, cfg, device) # [num_grids, num_ws, w_dim]
        trajectory = generate_camera_angles(cfg.camera)[0][:, :2] # [num_poses, 2]
        images = generate_trajectory(cfg, G, ws, trajectory) # [num_traj_steps, num_samples, c, h, w]
        images = images.permute(1, 0, 2, 3, 4) # [num_samples, num_traj_steps, c, h, w]

        for seed, grid in tqdm(list(zip(seeds, images)), desc='Saving'):
            # grid = TVF.resize(grid, size=(256, 256))
            grid = make_grid(grid, nrow=len(trajectory))
            TVF.to_pil_image(grid).save(os.path.join(save_dir, f'seed-{seed:04d}.jpg'), q=95)
    else:
        raise NotImplementedError(f"Unknown vis_type: {cfg.vis.name}")

#----------------------------------------------------------------------------

def sample_z_from_seeds(seeds: List[int], z_dim: int) -> torch.Tensor:
    zs = [np.random.RandomState(s).randn(1, z_dim) for s in seeds] # [num_samples, z_dim]
    return torch.from_numpy(np.concatenate(zs, axis=0)).float() # [num_samples, z_dim]

#----------------------------------------------------------------------------

def sample_c_from_seeds(seeds: List[int], c_dim: int) -> torch.Tensor:
    if c_dim == 0:
        return torch.empty(len(seeds), 0)

    c_idx = [np.random.RandomState(s).choice(np.arange(c_dim), size=1) for s in seeds] # [num_samples, 1]
    c_idx = np.concatenate(c_idx, axis=0) # [num_samples, 1]
    cs = np.zeros((len(seeds), c_dim)) # [num_samples, c_dim]
    cs[np.arange(len(seeds)), c_idx] = 1.0

    return torch.from_numpy(cs).float() # [num_samples, c_dim]

#----------------------------------------------------------------------------

def sample_ws_from_seeds(G, seeds: List[int], cfg: DictConfig, device: str, num_interp_steps: int=0):
    if num_interp_steps == 0:
        z = sample_z_from_seeds(seeds, G.z_dim).to(device) # [num_samples, z_dim]
        c = sample_c_from_seeds(seeds, G.c_dim).to(device) # [num_samples, c_dim]
        ws = G.mapping(z, c=c, truncation_psi=cfg.truncation_psi) # [num_samples, num_ws, w_dim]
    else:
        z_from = sample_z_from_seeds(seeds[0::2], G.z_dim).to(device) # [num_samples, z_dim]
        z_to = sample_z_from_seeds(seeds[1::2], G.z_dim).to(device) # [num_samples, z_dim]
        c_from = sample_c_from_seeds(seeds[0::2], G.c_dim).to(device) # [num_samples, c_dim]
        c_to = sample_c_from_seeds(seeds[1::2], G.c_dim).to(device) # [num_samples, c_dim]
        ws_from = G.mapping(z_from, c=c_from, truncation_psi=cfg.truncation_psi) # [num_samples, num_ws, w_dim]
        ws_to = G.mapping(z_to, c=c_to, truncation_psi=cfg.truncation_psi) # [num_samples, num_ws, w_dim]
        alpha = torch.linspace(0, 1, num_interp_steps, device=device).view(num_interp_steps, 1, 1, 1) # [num_interp_steps]
        ws = ws_from.unsqueeze(0) * (1 - alpha) + ws_to.unsqueeze(0) * alpha # [num_interp_steps, num_samples, num_ws, w_dim]

    return ws

#----------------------------------------------------------------------------

def generate_density(cfg, G, ws):
    coords = create_voxel_coords(cfg.volume_res, cfg.voxel_origin, cfg.cube_size, 1) # [batch_size, volume_res ** 3, 3]
    coords = coords.to(ws.device) # [batch_size, volume_res ** 3, 3]
    densities = []

    for idx in tqdm(range(len(ws))):
        curr_ws = ws[[idx]] # [1, num_ws, w_dim]
        sigma = G.synthesis.compute_densities(curr_ws, coords, max_batch_res=cfg.max_batch_res) # [batch_size, volume_res ** 3, 1]
        sigma = sigma.reshape(cfg.volume_res, cfg.volume_res, cfg.volume_res).cpu().numpy() # [volume_res ** 3]
        densities.append(sigma)

    return np.stack(densities)

#----------------------------------------------------------------------------

def create_voxel_coords(resolution=256, voxel_origin=[0.0, 0.0, 0.0], cube_size=2.0, batch_size: int=1):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_size / 2
    voxel_size = cube_size / (resolution - 1)

    overall_index = torch.arange(0, resolution ** 3, 1, out=torch.LongTensor())
    coords = torch.zeros(resolution ** 3, 3) # [h, w, d, 3]

    # transform first 3 columns
    # to be the x, y, z index
    coords[:, 2] = overall_index % resolution
    coords[:, 1] = (overall_index.float() / resolution) % resolution
    coords[:, 0] = ((overall_index.float() / resolution) / resolution) % resolution

    # transform first 3 columns
    # to be the x, y, z coordinate
    coords[:, 0] = (coords[:, 0] * voxel_size) + voxel_origin[2] # [volume_res ** 3]
    coords[:, 1] = (coords[:, 1] * voxel_size) + voxel_origin[1] # [volume_res ** 3]
    coords[:, 2] = (coords[:, 2] * voxel_size) + voxel_origin[0] # [volume_res ** 3]

    return coords.repeat(batch_size, 1, 1) # [batch_size, volume_res ** 3, 3]

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_vis() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

