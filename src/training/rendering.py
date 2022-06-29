"""
Volumetric rendering utils from pi-GAN generator
Adapted from https://github.com/marcoamonteiro/pi-GAN
"""

import math
import random
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.torch_utils import misc

#----------------------------------------------------------------------------

def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res

#----------------------------------------------------------------------------

def normalize(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return x / (torch.norm(x, dim=dim, keepdim=True))

#----------------------------------------------------------------------------

@misc.profiled_function
def fancy_integration(rgb_sigma, z_vals, noise_std=0.5, last_back=False, white_back_end_idx: int=0, clamp_mode=None, fill_mode=None, sp_beta: float=1.0, use_inf_depth: bool=True):
    """
    Performs NeRF volumetric rendering over features or colors.
    Assumes that the last dimension is density.

    rgb_sigma: [batch_size, h * w, num_steps, num_feats + 1] --- features to integrate
    z_vals: [batch_size, h * w, num_steps, num_feats + 1] --- z-values
    """
    rgbs = rgb_sigma[..., :-1] # [batch_size, h * w, num_steps, num_feats]
    sigmas = rgb_sigma[..., [-1]] # [batch_size, h * w, num_steps, 1]

    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1] # [batch_size, h * w, num_steps - 1, 1]
    deltas_last = (1e10 if use_inf_depth else 1e-3) * torch.ones_like(deltas[:, :, [0]]) # [batch_size, h * w, 1, 1]
    deltas = torch.cat([deltas, deltas_last], dim=2) # [batch_size, h * w, num_steps, 1]

    if noise_std > 0:
        sigmas = sigmas + noise_std * torch.randn_like(sigmas) # [batch_size, h * w, num_steps, 1]

    if clamp_mode == 'softplus':
        alphas = 1.0 - torch.exp(-deltas * F.softplus(sigmas, beta=sp_beta)) # [batch_size, h * w, num_steps, 1]
    elif clamp_mode == 'relu':
        alphas = 1.0 - torch.exp(-deltas * F.relu(sigmas)) # [batch_size, h * w, num_steps, 1]
    else:
        raise NotImplementedError(f"Uknown clamp mode: {clamp_mode}")

    # if use_inf_depth:
    #     alphas[:, :, -1, :] = 1.0 # [batch_size, h * w, ., 1]

    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, [0], :]), 1.0 - alphas + 1e-10], dim=2) # [batch_size, h * w, num_steps, 1]
    transmittance = torch.cumprod(alphas_shifted, dim=2)[:, :, :-1] # [batch_size, h * w, num_steps, 1]
    weights = alphas * transmittance # [batch_size, h * w, num_steps, 1]
    weights_agg = weights.sum(dim=2) # [batch_size, h * w, 1]

    if last_back:
        weights[:, :, -1] += (1.0 - weights_agg) # [batch_size, h * w, 1]

    rgb_final = (weights * rgbs).sum(dim=2) # [batch_size, h * w, num_feats]
    depth = (weights * z_vals).sum(dim=2) # [batch_size, h * w, 1]

    if white_back_end_idx > 0:
        # Using white back for the first `white_back_end_idx` channels
        # (it's typically set to the number of image channels)
        rgb_final[:, :, :white_back_end_idx] = rgb_final[:, :, :white_back_end_idx] + 1 - weights_agg

    if fill_mode == 'debug':
        num_colors = rgbs.shape[-1]
        red_color = torch.zeros(num_colors, device=rgb_sigma.device)
        red_color[0] = 1.0
        rgb_final[weights_agg.squeeze(-1) < 0.9] = red_color
    elif fill_mode == 'weight':
        rgb_final = weights_agg.expand_as(rgb_final)

    return {
        'rendered_feats': rgb_final,
        'depth': depth,
        'weights': weights,
        'final_transmittance': transmittance[:, :, [-1], :].squeeze(3).squeeze(2), # [batch_size, h * w]
    }

#----------------------------------------------------------------------------

def get_initial_rays_trig(batch_size: int, num_steps: int, device, fov: float, resolution: Tuple[int, int], ray_start: float, ray_end: float, patch_params: Dict=None):
    """
    Returns sample points, z_vals, and ray directions in camera space.

    If patch_scales/patch_offsets (of shape [batch_size, 2] each, for [0, 1] range) are provided,
    then will rescale the x/y plane accordingly to shoot rays into the desired region
    """
    compute_batch_size = 1 if (patch_params is None and type(fov) is float) else batch_size # Batch size used for computations
    w, h = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, w, device=device), torch.linspace(1, -1, h, device=device), indexing='ij')
    x = x.T.flatten().unsqueeze(0).repeat(compute_batch_size, 1) # [compute_batch_size, h * w]
    y = y.T.flatten().unsqueeze(0).repeat(compute_batch_size, 1) # [compute_batch_size, h * w]

    if not patch_params is None:
        patch_scales, patch_offsets = patch_params['scales'], patch_params['offsets']
        misc.assert_shape(patch_scales, [batch_size, 2])
        misc.assert_shape(patch_offsets, [batch_size, 2])
        # First, shift [-1, 1] range into [0, 2]
        # Then, multiply by the patch size
        # After that, shift back to [-1, 1]
        # Finally, apply the offset (converted from [0, 1] to [0, 2])
        x = (x + 1.0) * patch_scales[:, 0].view(batch_size, 1) - 1.0 + patch_offsets[:, 0].view(batch_size, 1) * 2.0 # [compute_batch_size, h * w]
        y = (y + 1.0) * patch_scales[:, 1].view(batch_size, 1) - 1.0 + patch_offsets[:, 1].view(batch_size, 1) * 2.0 # [compute_batch_size, h * w]

    fov = fov if isinstance(fov, torch.Tensor) else torch.tensor([fov], device=device) # [compute_batch_size]
    fov_rad = fov.unsqueeze(1).expand(compute_batch_size, 1) / 360 * 2 * np.pi # [compute_batch_size, 1]
    z = -torch.ones((compute_batch_size, h * w), device=device) / torch.tan(fov_rad * 0.5) # [compute_batch_size, h * w]
    rays_d_cam = normalize(torch.stack([x, y, z], dim=2), dim=2) # [compute_batch_size, h * w, 3]
    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device) # [num_steps]
    z_vals = z_vals.reshape(1, 1, num_steps, 1).repeat(compute_batch_size, h * w, 1, 1) # [1, h * w, num_steps, 1]

    if compute_batch_size == 1:
        z_vals = z_vals.repeat(batch_size, 1, 1, 1) # [batch_size, h * w, num_steps, 1]
        rays_d_cam = rays_d_cam.repeat(batch_size, 1, 1) # [batch_size, h * w, 3]

    return z_vals, rays_d_cam

#----------------------------------------------------------------------------

def perturb_points(z_vals):
    mids = 0.5 * (z_vals[:, :, 1:, :] + z_vals[:, :, :-1, :]) # [batch_size, h * w, num_steps - 1, 1]
    upper = torch.cat([mids, z_vals[:, :, -1:, :]], dim=2) # [batch_size, h * w, num_steps, 1]
    lower = torch.cat([z_vals[:, :, :1, :], mids], dim=2) # [batch_size, h * w, num_steps, 1]
    noise = torch.rand_like(z_vals) # [batch_size, h * w, num_steps, 1]
    z_vals = lower + (upper - lower) * noise # [batch_size, h * w, num_steps, 1]

    return z_vals

#----------------------------------------------------------------------------

@misc.profiled_function
def transform_points(z_vals, ray_directions, c2w: torch.Tensor, perturb: bool=True):
    """
    Samples a camera position and maps points in the camera space to the world space.
    points: [batch_size, h * w, num_steps, ?]
    c2w: camera-to-world matrix
    """
    batch_size, num_rays, num_steps, _ = z_vals.shape
    if perturb:
        z_vals = perturb_points(z_vals)
    points_homogeneous = torch.ones((batch_size, num_rays, num_steps, 4), device=z_vals.device)
    points_homogeneous[:, :, :, :3] = z_vals * ray_directions.unsqueeze(2) # [batch_size, h * w, num_steps, 3]

    # should be batch_size x 4 x 4 , batch_size x r^2 x num_steps x 4
    points_world = torch.bmm(c2w, points_homogeneous.reshape(batch_size, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(batch_size, num_rays, num_steps, 4)
    ray_d_world = torch.bmm(c2w[..., :3, :3], ray_directions.reshape(batch_size, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(batch_size, num_rays, 3)

    homogeneous_origins = torch.zeros((batch_size, 4, num_rays), device=z_vals.device)
    homogeneous_origins[:, 3, :] = 1
    ray_o_world = torch.bmm(c2w, homogeneous_origins).permute(0, 2, 1).reshape(batch_size, num_rays, 4)[..., :3]

    return points_world[..., :3], z_vals, ray_d_world, ray_o_world

#----------------------------------------------------------------------------

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

#----------------------------------------------------------------------------

def sample_camera_angles(cfg: Dict, batch_size: int, device: str):
    """
    Samples batch_size random locations along a sphere. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """

    horizontal_stddev = cfg['horizontal_stddev']
    vertical_stddev = cfg['vertical_stddev']
    horizontal_mean = cfg['horizontal_mean']
    vertical_mean = cfg['vertical_mean']
    mode = cfg['dist']

    if mode == 'uniform':
        theta = (torch.rand((batch_size, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((batch_size, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean
    elif mode in ('normal', 'gaussian'):
        theta = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((batch_size, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((batch_size, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
    elif mode == 'truncated_gaussian':
        theta = truncated_normal_(torch.zeros((batch_size, 1), device=device)) * horizontal_stddev + horizontal_mean
        phi = truncated_normal_(torch.zeros((batch_size, 1), device=device)) * vertical_stddev + vertical_mean
    elif mode == 'spherical_uniform':
        theta = (torch.rand((batch_size, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = ((torch.rand((batch_size,1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)
    else:
        raise NotImplementedError(f'Unknown distribution: {mode}')
        # Just use the mean.
        theta = torch.ones((batch_size, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((batch_size, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)
    angles = torch.cat([theta, phi, torch.zeros_like(theta)], dim=1) # [batch_size, 3]

    return angles

#----------------------------------------------------------------------------

def compute_camera_origins(angles: torch.Tensor, radius: float) -> torch.Tensor:
    misc.assert_shape(angles, [None, 3])

    yaw = angles[:, [0]] # [batch_size, 1]
    pitch = angles[:, [1]] # [batch_size, 1]

    assert yaw.ndim == 2, f"Wrong shape: {yaw.shape}, {pitch.shape}"
    assert yaw.shape == pitch.shape, f"Wrong shape: {yaw.shape}, {pitch.shape}"

    origins = torch.zeros((yaw.shape[0], 3), device=yaw.device)
    origins[:, [0]] = radius * torch.sin(pitch) * torch.cos(yaw)
    origins[:, [2]] = radius * torch.sin(pitch) * torch.sin(yaw)
    origins[:, [1]] = radius * torch.cos(pitch)

    return origins

#----------------------------------------------------------------------------

def compute_cam2world_matrix(camera_angles: torch.Tensor, radius: float):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""
    camera_origins = compute_camera_origins(camera_angles, radius) # [batch_size, 3]
    forward_vector = normalize(-camera_origins) # [batch_size, 3]
    batch_size = forward_vector.shape[0]
    forward_vector = normalize(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=forward_vector.device).expand_as(forward_vector)
    left_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=forward_vector.device).unsqueeze(0).repeat(batch_size, 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=forward_vector.device).unsqueeze(0).repeat(batch_size, 1, 1)
    translation_matrix[:, :3, 3] = camera_origins

    cam2world = translation_matrix @ rotation_matrix

    return cam2world

#----------------------------------------------------------------------------

@misc.profiled_function
def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) --- padded to [0, 1] inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]

    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)
    denom[denom < eps] = 1

    samples = bins_g[..., 0] + (u - cdf_g[...,0]) / denom * (bins_g[..., 1] - bins_g[..., 0])

    return samples

#----------------------------------------------------------------------------

def validate_frustum(fov: float, near: float, far: float, radius: float, scale: float=1.0, step: float=1e-2, device: str='cpu', verbose: bool=False) -> bool:
    """
    Generates a lot of points on a hemisphere of radius `radius`,
    computes the corners of the viewing frustum
    and checks that all these corners are inside the [-1, 1]^3 cube
    """
    # Step 1: sample the angles
    num_angles = int((np.pi / 2) / step) # [1]
    yaw = torch.linspace(0, np.pi * 2, steps=num_angles, device=device) # [num_angles]
    pitch = torch.linspace(0, np.pi, steps=num_angles, device=device) # [num_angles]
    yaw, pitch = torch.meshgrid(yaw, pitch, indexing='ij') # [num_angles, num_angles], [num_angles, num_angles]
    pitch = torch.clamp(pitch, 1e-7, np.pi - 1e-7)
    roll = torch.zeros(yaw.shape, device=device) # [num_angles, num_angles]
    angles = torch.stack([yaw.reshape(-1), pitch.reshape(-1), roll.reshape(-1)], dim=1) # [num_angles * num_angles, 3]
    batch_size = angles.shape[0]

    # Step 3: generating rays
    h = w = 2
    num_steps = 2
    x, y = torch.meshgrid(torch.linspace(-1, 1, w, device=device), torch.linspace(1, -1, h, device=device), indexing='ij')
    x = x.T.flatten().unsqueeze(0).repeat(batch_size, 1) # [batch_size, h * w]
    y = y.T.flatten().unsqueeze(0).repeat(batch_size, 1) # [batch_size, h * w]

    fov_rad = fov / 360 * 2 * np.pi # [1]
    z = -torch.ones((batch_size, h * w), device=device) / np.tan(fov_rad * 0.5) # [batch_size, h * w]
    rays_d_cam = normalize(torch.stack([x, y, z], dim=2), dim=2) # [batch_size, h * w, 3]

    z_vals = torch.linspace(near, far, num_steps, device=device).reshape(1, 1, num_steps, 1).repeat(batch_size, h * w, 1, 1) # [batch_size, h * w, num_steps, 1]
    c2w = compute_cam2world_matrix(camera_angles=angles, radius=radius) # [batch_size, 4, 4]
    points_world, _, _, _ = transform_points(z_vals=z_vals, ray_directions=rays_d_cam, c2w=c2w)

    if verbose:
        print('min/max coordinates for the near plane', points_world[:, :, 0].min().item(), points_world[:, :, 0].max().item())
        print('min/max coordinates for the far plane', points_world[:, :, 1].min().item(), points_world[:, :, 1].max().item())
        print('min/max coordinates total', points_world.min().item(), points_world.max().item())

    return points_world.min().item() >= -scale and points_world.max().item() <= scale

#----------------------------------------------------------------------------

def get_euler_angles(T: np.ndarray) -> Tuple[float, float, float]:
    yaw = np.arctan2(T[1, 0], T[0, 0]).item()
    pitch = np.arctan2(T[2, 1], T[2, 2]).item()
    if pitch < 0:
        assert pitch < 1e-8 or (np.pi + pitch) < 1e-8, f"Cannot handle pitch value: {pitch}"
        pitch = abs(pitch)
    roll = np.arctan2(-T[2, 0], np.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2)).item()

    return yaw, pitch, roll

#----------------------------------------------------------------------------

def compute_bg_points(ray_o_world: torch.Tensor, ray_d_world: torch.Tensor, num_steps: int, use_noise: float=False, bg_start: float=1.0):
    """
    ray_o_world: [batch_size, num_pixels, 3] --- ray origins
    ray_d_world: [batch_size, num_pixels, 3] --- ray directions (possibly unnormalized)
    bg_start: when should the background start after the sphere of radius 1.0 in terms of inverse depth
    """
    batch_size, num_pixels = ray_d_world.shape[:2]

    bg_z_vals = torch.linspace(bg_start, 0.0, steps=num_steps).to(ray_d_world.device) # [num_steps], "1 -> 0" format
    bg_z_vals = bg_z_vals.repeat(batch_size, num_pixels, 1) # [batch_size, num_pixels, num_steps]
    bg_z_vals = add_noise_to_interval(bg_z_vals) if use_noise else bg_z_vals # [batch_size, num_pixels, num_steps]

    ray_d_world_norm = ray_d_world / ray_d_world.norm(dim=2, keepdim=True) # [batch_size, num_pixels]
    ray_o_world = ray_o_world.unsqueeze(2).expand(batch_size, num_pixels, num_steps, 3) # [batch_size, num_pixels, num_steps, 3]
    ray_d_world_norm = ray_d_world_norm.unsqueeze(2).expand(batch_size, num_pixels, num_steps, 3) # [batch_size, num_pixels, num_steps, 3]
    bg_pts, _ = depth2pts_outside(ray_o_world, ray_d_world_norm, bg_z_vals) # [batch_size, num_pixels, num_steps, 4]
    bg_pts = bg_pts.reshape(batch_size, -1, 4) # [batch_size, num_pixels * num_steps, 4]
    ray_d_world_norm = ray_d_world_norm.reshape(batch_size, -1, 3) # [batch_size, num_pixels * num_steps, 3]

    return bg_pts, bg_z_vals, ray_d_world_norm

#----------------------------------------------------------------------------

def add_noise_to_interval(di):
    """
    Copy-pasted from https://github.com/facebookresearch/StyleNeRF
    """
    di_mid  = .5 * (di[..., 1:] + di[..., :-1])
    di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
    di_low  = torch.cat([di[..., :1], di_mid], dim=-1)
    noise   = torch.rand_like(di_low)
    ti      = di_low + (di_high - di_low) * noise
    misc.assert_shape(ti, di.shape)
    return ti

#----------------------------------------------------------------------------

def depth2pts_outside(ray_o: torch.Tensor, ray_d: torch.Tensor, depth: torch.Tensor):
    """
    Copy-pasted from https://github.com/Kai-46/nerfplusplus
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    """
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

    # Now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
        torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
        rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + 1e-6) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real

#----------------------------------------------------------------------------
