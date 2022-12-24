import math
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.torch_utils import misc
from src.torch_utils import persistence
from omegaconf import DictConfig

from src.training.networks_stylegan2 import SynthesisBlock
from src.training.networks_stylegan3 import SynthesisNetwork as SG3SynthesisNetwork
from src.training.networks_inr_gan import INR, nerf_renderer
from src.training.layers import (
    FullyConnectedLayer,
    MappingNetwork,
    ScalarEncoder1d,
)
from src.training.rendering import (
    fancy_integration,
    get_initial_rays_trig,
    transform_points,
    sample_pdf,
    compute_cam2world_matrix,
    compute_bg_points,
)
from src.training.training_utils import linear_schedule, run_batchwise, extract_patches

#----------------------------------------------------------------------------

@misc.profiled_function
def tri_plane_renderer(x: torch.Tensor, coords: torch.Tensor, ray_d_world: torch.Tensor, mlp: Callable, scale: float=1.0) -> torch.Tensor:
    """
    Computes RGB\sigma values from a tri-plane representation + MLP

    x: [batch_size, feat_dim * 3, h, w]
    coords: [batch_size, h * w, num_steps, 3]
    ray_d_world: [batch_size, h * w, 3] --- ray directions in the world coordinate system
    mlp: additional transform to apply on top of features
    scale: additional scaling of the coordinates
    """
    assert x.shape[1] % 3 == 0, f"We use 3 planes: {x.shape}"
    coords = coords.view(coords.shape[0], -1, 3) # [batch_size, h * w * num_points, 3]
    batch_size, raw_feat_dim, h, w = x.shape
    num_points = coords.shape[1]
    feat_dim = raw_feat_dim // 3
    misc.assert_shape(coords, [batch_size, None, 3])

    x = x.view(batch_size * 3, feat_dim, h, w) # [batch_size * 3, feat_dim, h, w]
    coords = coords / scale # [batch_size, num_points, 3]
    coords_2d = torch.stack([
        coords[..., [0, 1]], # z/y plane
        coords[..., [0, 2]], # z/x plane
        coords[..., [1, 2]], # y/x plane
    ], dim=1) # [batch_size, 3, num_points, 2]
    coords_2d = coords_2d.view(batch_size * 3, 1, num_points, 2) # [batch_size * 3, 1, num_points, 2]
    # assert ((coords_2d.min().item() >= -1.0 - 1e-8) and (coords_2d.max().item() <= 1.0 + 1e-8))
    x = F.grid_sample(x, grid=coords_2d, mode='bilinear', align_corners=True).view(batch_size, 3, feat_dim, num_points) # [batch_size, 3, feat_dim, num_points]
    x = x.permute(0, 1, 3, 2) # [batch_size, 3, num_points, feat_dim]
    x = mlp(x, coords, ray_d_world) # [batch_size, num_points, out_dim]

    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class TriPlaneMLP(nn.Module):
    def __init__(self, cfg: DictConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim

        if self.cfg.tri_plane.mlp.n_layers == 0:
            assert self.cfg.tri_plane.feat_dim == (self.out_dim + 1), f"Wrong dims: {self.cfg.tri_plane.feat_dim}, {self.out_dim}"
            self.model = nn.Identity()
        else:
            if self.cfg.tri_plane.get('posenc_period_len', 0) > 0:
                self.pos_enc = ScalarEncoder1d(3, x_multiplier=self.cfg.tri_plane.posenc_period_len, const_emb_dim=0)
            else:
                self.pos_enc = None

            backbone_input_dim = self.cfg.tri_plane.feat_dim + (0 if self.pos_enc is None else self.pos_enc.get_dim())
            backbone_out_dim = 1 + (self.cfg.tri_plane.mlp.hid_dim if self.cfg.tri_plane.has_view_cond else self.out_dim)
            self.dims = [backbone_input_dim] + [self.cfg.tri_plane.mlp.hid_dim] * (self.cfg.tri_plane.mlp.n_layers - 1) + [backbone_out_dim] # (n_hid_layers + 2)
            activations = ['lrelu'] * (len(self.dims) - 2) + ['linear']
            assert len(self.dims) > 2, f"We cant have just a linear layer here: nothing to modulate. Dims: {self.dims}"
            layers = [FullyConnectedLayer(self.dims[i], self.dims[i+1], activation=a) for i, a in enumerate(activations)]
            self.model = nn.Sequential(*layers)

            if self.cfg.tri_plane.has_view_cond:
                self.ray_dir_enc = ScalarEncoder1d(coord_dim=3, const_emb_dim=0, x_multiplier=8, use_cos=False, use_raw=True)
                self.color_network = nn.Sequential(
                    FullyConnectedLayer(backbone_out_dim - 1 + self.ray_dir_enc.get_dim(), 32, activation='lrelu'),
                    FullyConnectedLayer(32, self.out_dim, activation='linear'),
                )
            else:
                self.ray_dir_enc = None
                self.color_network = None

    def forward(self, x: torch.Tensor, coords: torch.Tensor, ray_d_world: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: [batch_size, 3, num_points, feat_dim] --- volumetric features from tri-planes
            coords: [batch_size, num_points, 3] --- coordinates, assumed to be in [-1, 1]
            ray_d_world: [batch_size, h * w, 3] --- camera ray's view directions
        """
        batch_size, _, num_points, feat_dim = x.shape
        x = x.mean(dim=1).reshape(batch_size * num_points, feat_dim) # [batch_size * num_points, feat_dim]
        if not self.pos_enc is None:
            misc.assert_shape(coords, [batch_size, num_points, 3])
            pos_embs = self.pos_enc(coords.reshape(batch_size * num_points, 3)) # [batch_size, num_points, pos_emb_dim]
            x = torch.cat([x, pos_embs], dim=1) # [batch_size, num_points, feat_dim + pos_emb_dim]
        x = self.model(x) # [batch_size * num_points, backbone_out_dim]
        x = x.view(batch_size, num_points, self.dims[-1]) # [batch_size, num_points, backbone_out_dim]

        if not self.color_network is None:
            num_pixels, view_dir_emb = ray_d_world.shape[1], self.ray_dir_enc.get_dim()
            num_steps = num_points // num_pixels
            ray_dir_embs = self.ray_dir_enc(ray_d_world.reshape(-1, 3)) # [batch_size * h * w, view_dir_emb]
            ray_dir_embs = ray_dir_embs.reshape(batch_size, num_pixels, 1, view_dir_emb) # [batch_size, h * w, 1, view_dir_emb]
            ray_dir_embs = ray_dir_embs.repeat(1, 1, num_steps, 1) # [batch_size, h * w, num_steps, view_dir_emb]
            ray_dir_embs = ray_dir_embs.reshape(batch_size, num_points, view_dir_emb) # [batch_size, h * w * num_steps, view_dir_emb]
            density = x[:, :, [-1]] # [batch_size, num_points, 1]
            color_feats = F.leaky_relu(x[:, :, :-1], negative_slope=0.1) # [batch_size, num_points, backbone_out_dim - 1]
            color_feats = torch.cat([color_feats, ray_dir_embs], dim=2) # [batch_size, num_points, backbone_out_dim - 1 + view_dir_emb]
            color_feats = color_feats.view(batch_size * num_points, self.dims[-1] - 1 + view_dir_emb) # [batch_size * num_points, backbone_out_dim - 1 + view_dir_emb]
            colors = self.color_network(color_feats) # [batch_size * num_points, out_dim]
            colors = colors.view(batch_size, num_points, self.out_dim) # [batch_size * num_points, out_dim]
            y = torch.cat([colors, density], dim=2) # [batch_size, num_points, out_dim + 1]
        else:
            y = x

        misc.assert_shape(y, [batch_size, num_points, self.out_dim + 1])

        return y

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlocksSequence(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        in_resolution,              # Which resolution do we start with?
        out_resolution,             # Output image resolution.
        in_channels,                # Number of input channels.
        out_channels,               # Number of input channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert in_resolution == 0 or (in_resolution >= 4 and math.log2(in_resolution).is_integer())
        assert out_resolution >= 4 and math.log2(out_resolution).is_integer()
        assert in_resolution < out_resolution

        super().__init__()

        self.w_dim = w_dim
        self.out_resolution = out_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fp16_res = num_fp16_res

        in_resolution_log2 = 2 if in_resolution == 0 else (int(np.log2(in_resolution)) + 1)
        out_resolution_log2 = int(np.log2(out_resolution))
        self.block_resolutions = [2 ** i for i in range(in_resolution_log2, out_resolution_log2 + 1)]
        out_channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (out_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for block_idx, res in enumerate(self.block_resolutions):
            cur_in_channels = out_channels_dict[res // 2] if block_idx > 0 else in_channels
            cur_out_channels = out_channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.out_resolution)
            block = SynthesisBlock(cur_in_channels, cur_out_channels, w_dim=w_dim, resolution=res,
                img_channels=self.out_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, x: torch.Tensor=None, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,            # Main config
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        num_fp16_res = 4,           # Number of FP16 res blocks for the upsampler
        **synthesis_seq_kwargs,     # Arguments of SynthesisBlocksSequence
    ):
        super().__init__()
        self.cfg = cfg
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        decoder_out_channels = self.cfg.tri_plane.feat_dim * 3 + (self.img_channels if self.cfg.bg_model.type == "plane" else 0)

        if self.cfg.backbone == 'stylegan2':
            self.tri_plane_decoder = SynthesisBlocksSequence(
                w_dim=w_dim,
                in_resolution=0,
                out_resolution=self.cfg.tri_plane.res,
                in_channels=0,
                out_channels=decoder_out_channels,
                architecture='skip',
                num_fp16_res=(0 if self.cfg.tri_plane.fp32 else num_fp16_res),
                use_noise=self.cfg.use_noise,
                **synthesis_seq_kwargs,
            )
        elif self.cfg.backbone == 'stylegan3-r':
            self.tri_plane_decoder = SG3SynthesisNetwork(
                w_dim=w_dim,
                img_resolution=self.cfg.tri_plane.res,
                img_channels=decoder_out_channels,
                num_fp16_res=(0 if self.cfg.tri_plane.fp32 else num_fp16_res),
                **synthesis_seq_kwargs,
            )
        elif self.cfg.backbone == 'raw_planes':
            self.tri_plane_decoder = nn.Parameter(torch.randn(1, decoder_out_channels, self.cfg.tri_plane.res, self.cfg.tri_plane.res))
            self.tri_plane_decoder.num_ws = 1
        else:
            raise NotImplementedError(f'Uknown backbone: {self.cfg.backbone}')

        self.tri_plane_mlp = TriPlaneMLP(self.cfg, out_dim=self.img_channels)
        self.num_ws = self.tri_plane_decoder.num_ws
        self.nerf_noise_std = 0.0
        self.train_resolution = self.cfg.patch.resolution if self.cfg.patch.enabled else self.img_resolution
        self.test_resolution = self.img_resolution

        if self.cfg.bg_model.type in (None, "plane"):
            self.bg_model = None
        elif self.cfg.bg_model.type == "sphere":
            self.bg_model = INR(self.cfg.bg_model, w_dim)
            self.num_ws += self.bg_model.num_ws
        else:
            raise NotImplementedError(f"Uknown BG model type: {self.bg_model}")

    def progressive_update(self, cur_kimg: float):
        self.nerf_noise_std = linear_schedule(cur_kimg, self.cfg.nerf_noise_std_init, 0.0, self.cfg.nerf_noise_kimg_growth)

    @torch.no_grad()
    def compute_densities(self, ws: torch.Tensor, coords: torch.Tensor, max_batch_res: int=32, use_bg: bool=False, **block_kwargs) -> torch.Tensor:
        """
        coords: [batch_size, num_points, 3]
        """
        assert not use_bg, f"Background NeRF is not supported."
        if self.cfg.backbone == 'raw_planes':
            plane_feats = self.tri_plane_decoder.repeat(len(ws), 1, 1, 1) + ws.sum() * 0.0 # [batch_size, 3, 256, 256]
        else:
            plane_feats = self.tri_plane_decoder(ws[:, :self.tri_plane_decoder.num_ws], **block_kwargs) # [batch_size, 3 * feat_dim, tp_h, tp_w]
        ray_d_world = torch.zeros_like(coords) # [batch_size, num_points, 3]
        output = run_batchwise(
            fn=tri_plane_renderer, data=dict(coords=coords, ray_d_world=ray_d_world),
            batch_size=max_batch_res ** 3,
            dim=1, mlp=self.tri_plane_mlp, x=plane_feats, scale=self.cfg.dataset.cube_scale,
        ) # [batch_size, num_coords, num_feats]

        return output[:, :, -1:] # [batch_size, num_coords, 1]

    def render_spherical_bg(self, ws_bg: torch.Tensor, ray_o_world: torch.Tensor, ray_d_world: torch.Tensor, max_batch_res: int) -> torch.Tensor:
        batch_size, num_pixels, _ = ray_o_world.shape
        white_back_end_idx = self.img_channels if self.cfg.dataset.white_back else 0

        # Adding the points from background NeRF
        bg_points, bg_z_vals, ray_d_world_norm = compute_bg_points(
            ray_o_world=ray_o_world,
            ray_d_world=ray_d_world,
            num_steps=self.cfg.bg_model.num_steps,
            use_noise=self.training,
            bg_start=self.cfg.bg_model.start,
        ) # [batch_size, num_pixels * num_bg_steps, 4], [batch_size, num_pixels, num_bg_steps], [batch_size, num_pixels, num_bg_steps, 3]
        bg_z_vals = bg_z_vals.view(batch_size, num_pixels * self.cfg.bg_model.num_steps, 1) # [batch_size, h * w * num_bg_steps, 1]
        ray_d_world_norm = ray_d_world_norm.reshape(batch_size, num_pixels * self.cfg.bg_model.num_steps, 3) # [batch_size, num_pixels * num_bg_steps, 3]

        # Evaluate the background NeRF
        bg_output = run_batchwise(
            fn=nerf_renderer,
            data=dict(coords=bg_points, ray_d_world=ray_d_world_norm),
            batch_size=max_batch_res ** 2 * self.cfg.bg_model.num_steps,
            dim=1, nerf=self.bg_model, ws=ws_bg,
        ) # [batch_size, h * w * num_bg_steps, num_feats]

        bg_output = bg_output.view(batch_size, num_pixels, self.cfg.bg_model.num_steps, self.cfg.bg_model.output_channels) # [batch_size, h * w, num_bg_steps, 4]
        bg_z_vals = bg_z_vals.view(batch_size, num_pixels, self.cfg.bg_model.num_steps, 1) # [batch_size, h * w, num_bg_steps, 1]

        # Perform volumetric rendering
        bg_int_out: Dict = run_batchwise(
            fn=fancy_integration,
            data=dict(rgb_sigma=bg_output, z_vals=-bg_z_vals), # bg_z_vals are in the "1 -> 0" format, so negate it to get "-1 -> 0"
            batch_size=max_batch_res ** 2,
            dim=1,
            white_back_end_idx=white_back_end_idx, last_back=False, use_inf_depth=True,
            clamp_mode=self.cfg.clamp_mode, noise_std=0.0)

        return bg_int_out['rendered_feats'] # [batch_size, h * w, mlp_out_dim]

    def forward(self, ws, camera_angles: torch.Tensor, patch_params: Dict=None, max_batch_res: int=128, return_depth: bool=False, ignore_bg: bool=False, bg_only: bool=False, fov=None, **block_kwargs):
        """
        ws: [batch_size, num_ws, w_dim] --- latent codes
        camera_angles: [batch_size, 3] --- yaw/pitch/roll angles (roll angles are never used)
        patch_params: Dict {scales: [batch_size, 2], offsets: [batch_size, 2]} --- patch parameters (when we do patchwise training)
        """
        misc.assert_shape(camera_angles, [len(ws), 3])

        if self.cfg.backbone == 'raw_planes':
            plane_feats = self.tri_plane_decoder.repeat(len(ws), 1, 1, 1) + ws.sum() * 0.0 # [batch_size, 3, 256, 256]
        else:
            plane_feats = self.tri_plane_decoder(ws[:, :self.tri_plane_decoder.num_ws], **block_kwargs) # [batch_size, 3 * feat_dim, tp_h, tp_w]

        camera_angles[:, [1]] = torch.clamp(camera_angles[:, [1]], 1e-5, np.pi - 1e-5) # [batch_size, 1]
        batch_size = ws.shape[0]
        h = w = (self.train_resolution if self.training else self.test_resolution)
        should_render_bg = (not self.cfg.bg_model.type is None) and (not ignore_bg)
        fov = self.cfg.dataset.camera.fov if fov is None else fov # [1] or [batch_size]

        if self.cfg.bg_model.type == "plane":
            bg = plane_feats[:, -self.img_channels:, :, :] # [batch_size, h, w]
            bg = F.interpolate(bg, size=(self.img_resolution, self.img_resolution), mode='bilinear', align_corners=True) # [batch_size, c, h, w]
            plane_feats = plane_feats[:, :-self.img_channels, :, :].contiguous() # [batch_size, c, h, w]

            if not patch_params is None:
                bg = extract_patches(bg, patch_params, resolution=self.cfg.patch.resolution) # [batch_size, c, h_patch, w_patch]
            bg = bg.view(batch_size, self.img_channels, h * w).permute(0, 2, 1) # # [batch_size, h * w, c]

        num_steps = self.cfg.num_ray_steps
        tri_plane_out_dim = self.img_channels + 1
        white_back_end_idx = self.img_channels if self.cfg.dataset.white_back else 0
        nerf_noise_std = self.nerf_noise_std if self.training else 0.0

        z_vals, rays_d_cam = get_initial_rays_trig(
            batch_size, num_steps, resolution=(h, w), device=ws.device, ray_start=self.cfg.dataset.camera.ray_start,
            ray_end=self.cfg.dataset.camera.ray_end, fov=fov, patch_params=patch_params)
        c2w = compute_cam2world_matrix(camera_angles, self.cfg.dataset.camera.radius) # [batch_size, 4, 4]
        points_world, z_vals, ray_d_world, ray_o_world = transform_points(z_vals=z_vals, ray_directions=rays_d_cam, c2w=c2w) # [batch_size, h * w, num_steps, 1], [?], [?], [batch_size, h * w, 3]

        coarse_output = run_batchwise(
            fn=tri_plane_renderer, data=dict(coords=points_world, ray_d_world=ray_d_world),
            batch_size=max_batch_res ** 2,
            dim=1, mlp=self.tri_plane_mlp, x=plane_feats, scale=self.cfg.dataset.cube_scale,
        ) # [batch_size, h * w * num_steps, num_feats]
        coarse_output = coarse_output.view(batch_size, h * w, num_steps, tri_plane_out_dim) # [batch_size, h * w, num_steps, num_feats]

        # <==================================================>
        # HIERARCHICAL SAMPLING START
        points_world = points_world.reshape(batch_size, h * w, num_steps, 3) # [batch_size, h * w, num_steps, 3]
        weights = run_batchwise(
            fn=fancy_integration,
            data=dict(rgb_sigma=coarse_output, z_vals=z_vals),
            batch_size=max_batch_res ** 2,
            dim=1,
            clamp_mode=self.cfg.clamp_mode, noise_std=nerf_noise_std, use_inf_depth=self.cfg.bg_model.type is None,
        )['weights'] # [batch_size, h * w, num_steps, 1]
        weights = weights.reshape(batch_size * h * w, num_steps) + 1e-5 # [batch_size * h * w, num_steps]

        # <= Importance sampling START =>
        z_vals = z_vals.reshape(batch_size * h * w, num_steps) # [batch_size * h * w, num_steps]
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:]) # [batch_size * h * w, num_steps - 1]
        z_vals = z_vals.reshape(batch_size, h * w, num_steps, 1) # [batch_size, h * w, num_steps, 1]
        fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
        fine_z_vals = fine_z_vals.reshape(batch_size, h * w, num_steps, 1)
        fine_points = ray_o_world.unsqueeze(2).contiguous() + ray_d_world.unsqueeze(2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous() # [batch_size, h * w, num_steps, 1]
        # <= Importance sampling END =>

        # Model prediction on re-sampled find points
        fine_output = run_batchwise(
            fn=tri_plane_renderer, data=dict(coords=fine_points, ray_d_world=ray_d_world),
            batch_size=max_batch_res ** 2,
            dim=1, mlp=self.tri_plane_mlp, x=plane_feats, scale=self.cfg.dataset.cube_scale,
        ) # [batch_size, h * w * num_steps, num_feats]
        fine_output = fine_output.view(batch_size, h * w, num_steps, tri_plane_out_dim) # [batch_size, h * w, num_steps, tri_plane_out_dim]

        # Combine coarse and fine points
        all_outputs = torch.cat([fine_output, coarse_output], dim=2) # [batch_size, h * w, 2 * num_steps, tri_plane_out_dim + 1]
        all_z_vals = torch.cat([fine_z_vals, z_vals], dim=2) # [batch_size, h * w, 2 * num_steps, 1]
        _, indices = torch.sort(all_z_vals, dim=2) # [batch_size, h * w, 2 * num_steps, 1]
        all_z_vals = torch.gather(all_z_vals, dim=2, index=indices) # [batch_size, h * w, 2 * num_steps, 1]
        all_outputs = torch.gather(all_outputs, dim=2, index=indices.expand(-1, -1, -1, tri_plane_out_dim)) # [batch_size, h * w, 2 * num_steps, tri_plane_out_dim + 1]
        # HIERARCHICAL SAMPLING END
        # <==================================================>

        int_out: Dict = run_batchwise(
            fn=fancy_integration,
            data=dict(rgb_sigma=all_outputs, z_vals=all_z_vals),
            batch_size=max_batch_res ** 2,
            dim=1,
            white_back_end_idx=white_back_end_idx, last_back=self.cfg.dataset.last_back, clamp_mode=self.cfg.clamp_mode,
            noise_std=nerf_noise_std, use_inf_depth=self.cfg.bg_model.type is None)
        misc.assert_shape(int_out['final_transmittance'], [batch_size, h * w])

        if should_render_bg:
            if self.cfg.bg_model.type == "sphere":
                bg = self.render_spherical_bg(
                    ws_bg=ws[:, self.tri_plane_decoder.num_ws:self.tri_plane_decoder.num_ws + self.bg_model.num_ws], # [batch_size, bg_model.num_ws, w_dim]
                    ray_o_world=ray_o_world,
                    ray_d_world=ray_d_world,
                    max_batch_res=max_batch_res,
                ) # [batch_size, h * w, mlp_out_dim]
            elif self.cfg.bg_model.type == "plane":
                pass # Already rendered

            # Combine the results with the foreground NeRF
            if bg_only:
                int_out['rendered_feats'] = bg # [batch_size, h * w, mlp_out_dim]
            else:
                int_out['rendered_feats'] = int_out['rendered_feats'] + bg * int_out['final_transmittance'].unsqueeze(2) # [batch_size, h * w, mlp_out_dim]
        else:
            assert not bg_only

        img = int_out['depth' if return_depth else 'rendered_feats'] # [batch_size, h * w, 1 | mlp_out_dim]
        img = img.reshape(batch_size, h, w, img.shape[2]) # [batch_size, h, w, 1 | tri_plane_out_dim - 1]
        img = img.permute(0, 3, 1, 2).contiguous() # [batch_size, 1 | mlp_out_dim, h, w]

        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,            # Main config
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.cfg = cfg
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(cfg=cfg, w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, camera_raw_scalars=True, **mapping_kwargs)

    def progressive_update(self, cur_kimg: float):
        self.synthesis.progressive_update(cur_kimg)

    def forward(self, z, c, camera_angles, camera_angles_cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, camera_angles=camera_angles_cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, camera_angles=camera_angles, update_emas=update_emas, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------
