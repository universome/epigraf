"""
A better (INR-GAN-based) NeRF architecture
"""

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.torch_utils import misc
from src.torch_utils import persistence
from src.torch_utils.ops import bias_act, fma
from src.training.layers import FullyConnectedLayer, ScalarEncoder1d

#----------------------------------------------------------------------------

@misc.profiled_function
def fmm_modulate_linear(x: torch.Tensor, weight: torch.Tensor, mod_params: torch.Tensor, noise=None, activation: str="demod") -> torch.Tensor:
    """
    x: [batch_size, c_in]
    weight: [c_out, c_in]
    style: [batch_size, num_mod_params]
    noise: Optional[batch_size, 1]
    """
    batch_size, c_in, num_points = x.shape
    c_out, c_in = weight.shape
    rank = mod_params.shape[1] // (c_in + c_out)
    assert mod_params.shape[1] % (c_in + c_out) == 0
    assert activation in ('tanh', 'sigmoid', 'demod'), f"Unknown activation: {activation}"

    # Now, we need to construct a [c_out, c_in] matrix
    left_matrix = mod_params[:, :c_out * rank] # [batch_size, left_matrix_size]
    right_matrix = mod_params[:, c_out * rank:] # [batch_size, right_matrix_size]

    left_matrix = left_matrix.view(batch_size, c_out, rank) # [batch_size, c_out, rank]
    right_matrix = right_matrix.view(batch_size, rank, c_in) # [batch_size, rank, c_in]

    # Imagine, that the output of `self.affine` (in SynthesisLayer) is N(0, 1)
    # Then, std of weights is sqrt(rank). Converting it back to N(0, 1)
    modulation = left_matrix @ right_matrix / np.sqrt(rank) # [batch_size, c_out, c_in]

    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5

    W = weight.unsqueeze(0) * (modulation + 1.0) # [batch_size, c_out, c_in]
    if activation == "demod":
        W = W / (W.norm(dim=2, keepdim=True) + 1e-8) # [batch_size, c_out, c_in]
    W = W.to(dtype=x.dtype) # [batch_size, c_out, c_in]

    out = torch.bmm(W, x) # [batch_size, c_out, num_points]

    if not noise is None:
        out = out.add_(noise) # [batch_size, c_out, num_points]

    return out

#----------------------------------------------------------------------------

@misc.profiled_function
def style_modulate_linear(x: torch.Tensor, weight: torch.Tensor, styles: torch.Tensor, noise=None, demodulate: bool=True) -> torch.Tensor:
    """
    x: [batch_size, c_in, num_points]
    weight: [c_out, c_in]
    style: [batch_size, c_in]
    noise: Optional[batch_size, 1]
    """
    batch_size, c_in, num_points = x.shape
    c_out, c_in = weight.shape
    misc.assert_shape(styles, [batch_size, c_in])

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(c_in) / weight.norm(float('inf'), dim=1, keepdim=True)) # [c_out, c_in]
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate:
        w = weight.unsqueeze(0) # [1, c_out, c_in]
        w = w * styles.reshape(batch_size, 1, -1) # [batch_size, c_out, c_in]
        dcoefs = (w.square().sum(dim=2) + 1e-8).rsqrt().to(x.dtype).reshape(batch_size, -1, 1) # [batch_size, c_out, 1]

    # Execute by scaling the activations before and after the convolution.
    x = x * styles.to(x.dtype).reshape(batch_size, -1, 1) # [batch_size, num_points, c_out]
    x = x.permute(0, 2, 1) @ weight.to(x.dtype).t() # [batch_size, num_points, c_out]
    x = x.permute(0, 2, 1) # [batch_size, c_out, num_points]
    if demodulate:
        if not noise is None:
            x = fma.fma(x, dcoefs, noise.to(x.dtype))
        else:
            x = x * dcoefs
    elif noise is not None:
        x = x.add_(noise.to(x.dtype))
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        cfg: DictConfig = {},           # Main config
    ):
        super().__init__()

        self.cfg = cfg
        self.w_dim = w_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        if self.cfg.fmm.enabled:
            self.affine = FullyConnectedLayer(w_dim, (in_channels + out_channels) * self.cfg.fmm.rank, bias_init=0)
        else:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, gain: float=1.0):
        misc.assert_shape(x, [None, self.weight.shape[1], None])
        mod_params = self.affine(w) # [batch_size, (c_out + c_in) * rank]
        if self.cfg.fmm.enabled:
            x = fmm_modulate_linear(x=x, weight=self.weight, mod_params=mod_params, activation=self.cfg.fmm.activation) # [batch_size, d, num_points]
        else:
            x = style_modulate_linear(x=x, weight=self.weight, styles=mod_params) # [batch_size, d, num_points]
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x.unsqueeze(3), self.bias.to(x.dtype), act=self.activation, gain=self.act_gain * gain, clamp=act_clamp).squeeze(3) # [batch_size, d, num_points]
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'activation={self.activation:s}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisOutputLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, conv_clamp=None):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels)

    def forward(self, x, w):
        styles = self.affine(w) * self.weight_gain
        x = style_modulate_linear(x=x, weight=self.weight, styles=styles, demodulate=False) # [batch_size, c_in, num_points]
        x = bias_act.bias_act(x.unsqueeze(3), self.bias.to(x.dtype), clamp=self.conv_clamp).squeeze(3) # [batch_size, c_in, num_points]
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        img_channels,                       # Number of output color channels. TODO: better name for this argument
        is_last,                            # Is this the last block?
        coord_dim,                          # Number of input coordinates. Typically, it's just 3: x, y, z
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        cfg                 = {},           # Additional config
        **synth_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.num_layers = 0
        self.num_toout = 0
        self.coord_dim = coord_dim

        if in_channels == 0:
            self.scalar_enc = ScalarEncoder1d(self.coord_dim, x_multiplier=self.cfg.posenc_period_len, const_emb_dim=0)
            conv1_in_channels = self.scalar_enc.get_dim()
        else:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, conv_clamp=conv_clamp, cfg=cfg, **synth_kwargs)
            self.num_layers += 1
            conv1_in_channels = out_channels

        self.conv1 = SynthesisLayer(conv1_in_channels, out_channels, w_dim=w_dim, conv_clamp=conv_clamp, cfg=cfg, **synth_kwargs)
        self.num_layers += 1

        if is_last or architecture == 'skip':
            self.to_out = SynthesisOutputLayer(out_channels, img_channels, w_dim=w_dim, conv_clamp=conv_clamp)
            self.num_toout += 1

    def forward(self, x, out, ws, force_fp32=False, **synth_kwargs):
        misc.assert_shape(ws, [None, self.num_layers + self.num_toout, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32

        # Input.
        if self.in_channels == 0:
            batch_size, coord_dim, num_points = x.shape
            assert coord_dim == self.coord_dim, f"Wrong input shape: {x.shape}"
            x = self.scalar_enc(x.permute(0, 2, 1).reshape(batch_size * num_points, self.coord_dim)).to(dtype) # [batch_size * num_points, d]
            x = x.reshape(batch_size, num_points, self.scalar_enc.get_dim()).permute(0, 2, 1) # [batch_size, d, num_points]
        else:
            misc.assert_shape(x, [None, self.in_channels, None])
            x = x.to(dtype=dtype)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), **synth_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), **synth_kwargs)
            x = self.conv1(x, next(w_iter), gain=np.sqrt(0.5), **synth_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), **synth_kwargs)
            x = self.conv1(x, next(w_iter), **synth_kwargs)

        # To output.
        if out is not None:
            misc.assert_shape(out, [None, self.img_channels, None])

        if self.is_last or self.architecture == 'skip':
            y = self.to_out(x, next(w_iter))
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            out = out.add_(y) if out is not None else y

        assert x.dtype == dtype
        assert out is None or out.dtype == torch.float32
        return x, out

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self, cfg: DictConfig, w_dim: int, **block_kwargs):
        super().__init__()
        self.w_dim = w_dim
        self.cfg = cfg
        channels_dict = {block_idx: min(self.cfg.cbase // 2 ** (block_idx + 2), self.cfg.cmax) for block_idx in range(self.cfg.num_blocks)}
        fp16_start_idx = max(self.cfg.num_blocks - self.cfg.num_fp16_blocks, 1)
        self.num_ws = 0
        self.blocks = nn.ModuleList()
        for block_idx in range(self.cfg.num_blocks):
            in_channels = channels_dict[block_idx - 1] if block_idx > 0 else 0
            out_channels = channels_dict[block_idx]
            use_fp16 = (block_idx >= fp16_start_idx)
            is_last = (block_idx == self.cfg.num_blocks - 1)
            block = SynthesisBlock(
                in_channels, out_channels, w_dim=w_dim, img_channels=self.cfg.output_channels,
                is_last=is_last, use_fp16=use_fp16, cfg=cfg, coord_dim=self.cfg.coord_dim, **block_kwargs)
            self.num_ws += block.num_layers + (block.num_toout if is_last else 0)
            self.blocks.append(block)

    def forward(self, coords: torch.Tensor, ws, ray_d_world: torch.Tensor=None, **block_kwargs):
        misc.assert_shape(coords, [len(ws), self.cfg.coord_dim, None])
        assert ray_d_world is None, f"View direction conditioning is not supported :|"
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32) # [batch_size, num_ws, w_dim]
            w_idx = 0
            for block in self.blocks:
                block_ws.append(ws.narrow(1, w_idx, block.num_layers + block.num_toout))
                w_idx += block.num_layers
        x = coords # [batch_size, coord_dim, num_coords]
        out = None
        for block, cur_ws in zip(self.blocks, block_ws):
            x, out = block(x, out, cur_ws, **block_kwargs)
        return out

#----------------------------------------------------------------------------
