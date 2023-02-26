import os
from typing import List, Optional

import torch
import torchvision as tv
from omegaconf import DictConfig
import PIL.Image
from tqdm import tqdm
import numpy as np

from src import dnnlib
from src.training.rendering import sample_camera_angles

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, cfg, random_seed=0):
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    all_indices = list(range(len(training_set)))
    np.random.RandomState(random_seed).shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    batch = [training_set[i] for i in grid_indices]
    images = [b['image'] for b in batch]
    labels = [b['label'] for b in batch]
    if cfg.dataset.camera.dist == 'custom':
        camera_angles = [b['camera_angles'] for b in batch]
    else:
        camera_angles = sample_camera_angles(cfg=cfg.dataset.camera, batch_size=len(batch), device='cpu').numpy()
    return (gw, gh), np.stack(images), np.stack(labels), np.stack(camera_angles)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname, q=95)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname, q=95)

#----------------------------------------------------------------------------

def generate_videos(G: torch.nn.Module, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    num_videos = 9 if G.img_resolution >= 1024 else 16
    z, c = z[:num_videos], c[:num_videos], # [num_videos, z_dim], [num_videos, c_dim]
    camera_cfg = dnnlib.EasyDict({'name': 'front_circle', 'num_frames': 32, 'yaw_diff': 0.5, 'pitch_diff': 0.3, 'fov_diff': 1.0})
    vis_cfg = dnnlib.EasyDict({'max_batch_res': 64, 'batch_size': 8})
    angles, fovs = generate_camera_angles(camera_cfg, default_fov=G.cfg.dataset.camera.fov) # [num_frames, 3], [num_frames]
    ws = G.mapping(z=z, c=c) # [num_videos, num_ws, w_dim]
    images = generate_trajectory(vis_cfg, G, ws, angles[:, :2], fovs=fovs, verbose=False) # [num_frames, num_videos, c, h, w]
    images = images.permute(1, 0, 2, 3, 4) # [num_videos, num_frames, c, h, w]

    return images

#----------------------------------------------------------------------------

def save_videos(videos: torch.Tensor, save_path: os.PathLike):
    grids = [tv.utils.make_grid(vs, nrow=int(np.sqrt(videos.shape[0]))) for vs in videos.permute(1, 0, 2, 3, 4)] # [num_frames, c, gh, gw]
    video = (torch.stack(grids) * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
    tv.io.write_video(save_path, video, fps=25, video_codec='h264', options={'crf': '20'})

#----------------------------------------------------------------------------

def generate_trajectory(cfg, G, ws: torch.Tensor, trajectory: List, fovs: torch.Tensor=None, **generate_kwargs):
    """Produces frames for all `ws` for each trajectory step"""
    assert isinstance(trajectory, np.ndarray)
    num_cameras, num_samples = len(trajectory), len(ws) # [1], [1]
    trajectory = torch.from_numpy(trajectory).float().to(ws.device) # [num_steps, 2]
    angles = torch.cat([trajectory, torch.zeros_like(trajectory[:, [0]])], dim=1) # [num_cameras, 3]
    angles = angles.repeat_interleave(len(ws), dim=0) # [num_cameras * num_samples, 3]
    fovs = None if fovs is None else torch.from_numpy(fovs).float().to(ws.device).repeat_interleave(len(ws), dim=0) # None or [num_cameras * num_samples]
    ws = ws.repeat(num_cameras, 1, 1) # [num_samples * num_cameras, num_ws, w_dim]
    images = generate(cfg, G, ws=ws, angles=angles, fovs=fovs, **generate_kwargs) # [num_cameras * num_samples, c, h, w]
    images = images.reshape(num_cameras, num_samples, *images.shape[1:]) # [num_cameras, num_samples, c, h, w]

    return images

#----------------------------------------------------------------------------

def generate(cfg: DictConfig, G, ws: torch.Tensor, angles: torch.Tensor, fovs: torch.Tensor=None, verbose: bool=True, **synthesis_kwargs):
    assert len(ws) == len(angles), f"Wrong shapes: {ws.shape} vs {angles.shape}"
    max_batch_res_kwargs = {} if cfg.max_batch_res is None else dict(max_batch_res=cfg.max_batch_res)
    synthesis_kwargs = dict(return_depth=False, noise_mode='const', **max_batch_res_kwargs, **synthesis_kwargs)
    frames = []
    batch_indices = range(0, (len(ws) + cfg.batch_size - 1) // cfg.batch_size)
    batch_indices = tqdm(batch_indices, desc='Generating') if verbose else batch_indices
    for batch_idx in batch_indices:
        curr_slice = slice(batch_idx * cfg.batch_size, (batch_idx + 1) * cfg.batch_size)
        curr_ws, curr_angles = ws[curr_slice], angles[curr_slice] # [batch_size, num_ws, w_dim], [batch_size, 3]
        curr_fovs = G.cfg.dataset.camera.fov if fovs is None else fovs[curr_slice] # [1] or [batch_size]
        frame = G.synthesis(curr_ws, camera_angles=curr_angles, fov=curr_fovs, **synthesis_kwargs) # [batch_size, c, h, w]
        frame = frame.clamp(-1, 1).cpu() * 0.5 + 0.5 # [batch_size, c, h, w]
        frames.extend(frame)
    return torch.stack(frames) # [num_frames, c, h, w]

#----------------------------------------------------------------------------

def generate_camera_angles(camera, default_fov: Optional[float]=None):
    if camera.name == 'front_circle':
        assert not default_fov is None
        steps = np.linspace(0, 1, camera.num_frames)
        pitch = np.pi / 2 + camera.pitch_diff * np.cos(steps * 2 * np.pi) # [num_frames]
        yaw = camera.yaw_diff * np.sin(steps * 2 * np.pi) # [num_frames]
        fovs = default_fov + camera.fov_diff * np.sin(steps * 2 * np.pi) # [num_frames]
        angles = np.stack([yaw, pitch, np.zeros(camera.num_frames)], axis=1) # [num_frames, 3]
    elif camera.name == 'points':
        angles = np.stack([camera.yaws, np.ones(len(camera.yaws)) * camera.pitch, np.zeros(len(camera.yaws))], axis=1) # [num_angles, 3]
        fovs = None
    elif camera.name == 'wiggle':
        yaws = np.linspace(camera.yaw_left, camera.yaw_right, camera.num_frames) # [num_frames]
        pitches = camera.pitch_diff * np.cos(np.linspace(0, 1, camera.num_frames) * 2 * np.pi) + np.pi/2
        angles = np.stack([yaws, pitches, np.zeros(yaws.shape)], axis=1) # [num_frames, 3]
        fovs = None
    elif camera.name == 'line':
        yaws = np.linspace(camera.yaw_left, camera.yaw_right, camera.num_frames) # [num_frames]
        pitches = np.linspace(camera.pitch_left, camera.pitch_right, camera.num_frames) # [num_frames]
        angles = np.stack([yaws, pitches, np.zeros(yaws.shape)], axis=1) # [num_frames, 3]
        fov = default_fov if camera.fov is None else camera.fov # [1]
        fovs = np.array([fov]).repeat(camera.num_frames) # [num_frames]
    else:
        raise NotImplementedError(f'Unknown camera: {camera.name}')

    assert angles.shape[1] == 3, f"Wrong shape: {angles.shape}"

    return angles, fovs

#----------------------------------------------------------------------------
