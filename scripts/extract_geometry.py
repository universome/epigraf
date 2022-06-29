from omegaconf import DictConfig
# from ast import DictComp
# import plyfile
# import argparse
import torch
import numpy as np
# import skimage.measure
# import scipy
import mcubes
import trimesh
import mrcfile
import os
import hydra
from tqdm import tqdm

from scripts.utils import load_generator, set_seed, create_voxel_coords

#----------------------------------------------------------------------------

@hydra.main(config_path="../../configs/scripts", config_name="extract_geometry.yaml")
def extract_geometry(cfg: DictConfig):
    device = torch.device('cuda')
    G = load_generator(cfg.ckpt, verbose=cfg.verbose)[0].to(device).eval()

    for seed in tqdm(cfg.seeds, desc='Extracting geometry...'):
        set_seed(seed)
        batch_size = 1
        z = torch.randn(batch_size, G.z_dim, device=device) # [batch_size, z_dim]
        c = torch.zeros(batch_size, G.c_dim, device=device) # [batch_size, c_dim]
        assert G.c_dim == 0
        coords = create_voxel_coords(cfg.voxel_res, cfg.voxel_origin, cfg.cube_size, batch_size) # [batch_size, voxel_res ** 3, 3]
        coords = coords.to(z.device) # [batch_size, voxel_res ** 3, 3]
        ws = G.mapping(z, c, truncation_psi=cfg.truncation_psi, noise_mode='const') # [batch_size, num_ws, w_dim]
        sigma = G.synthesis.compute_densities(ws, coords, max_batch_res=cfg.max_batch_res) # [batch_size, voxel_res ** 3, 1]
        assert batch_size == 1
        sigma = sigma.reshape(cfg.voxel_res, cfg.voxel_res, cfg.voxel_res).cpu().numpy() # [voxel_res ** 3]

        print('sigma percentiles:', {q: np.percentile(sigma.reshape(-1), q) for q in [50.0, 90.0, 95.0, 97.5, 99.0, 99.5]})
        vertices, triangles = mcubes.marching_cubes(sigma, np.percentile(sigma, cfg.thresh_percentile))
        mesh = trimesh.Trimesh(vertices, triangles)
        os.makedirs('shapes', exist_ok=True)
        mesh.export(f'shapes/shape-{seed}.obj')

        # os.makedirs(cfg.output_dir, exist_ok=True)
        # with mrcfile.new_mmap(os.path.join(cfg.output_dir, f'{seed}.mrc'), overwrite=True, shape=sigma.shape, mrc_mode=2) as mrc:
        #     mrc.data[:] = sigma

#----------------------------------------------------------------------------

if __name__ == "__main__":
    extract_geometry() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------