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

from scripts.utils import load_generator, set_seed
from scripts.inference import create_voxel_coords, sample_z_from_seeds, sample_c_from_seeds

#----------------------------------------------------------------------------

@hydra.main(config_path="../configs/scripts", config_name="extract_geometry.yaml")
def extract_geometry(cfg: DictConfig):
    device = torch.device('cuda')
    G = load_generator(cfg.ckpt, verbose=cfg.verbose)[0].to(device).eval()

    set_seed(42) # To fix non-z randomization
    assert (not cfg.seeds is None) or (not cfg.num_seeds is None), "You must specify either `num_seeds` or `seeds`"
    seeds = cfg.seeds if cfg.num_seeds is None else np.arange(cfg.num_seeds)

    for seed in tqdm(seeds, desc='Extracting geometry...'):
        print('Seed', seed)
        batch_size = 1
        z = sample_z_from_seeds([seed], G.z_dim).to(device) # [batch_size, z_dim]
        c = sample_c_from_seeds([seed], G.c_dim).to(device) # [batch_size, c_dim]
        coords = create_voxel_coords(cfg.voxel_res, cfg.voxel_origin, cfg.cube_size, batch_size) # [batch_size, voxel_res ** 3, 3]
        coords = coords.to(z.device) # [batch_size, voxel_res ** 3, 3]
        ws = G.mapping(z, c, truncation_psi=cfg.truncation_psi) # [batch_size, num_ws, w_dim]
        sigma = G.synthesis.compute_densities(ws, coords, max_batch_res=cfg.max_batch_res, noise_mode='const') # [batch_size, voxel_res ** 3, 1]
        assert batch_size == 1
        sigma = sigma.reshape(cfg.voxel_res, cfg.voxel_res, cfg.voxel_res).cpu().numpy() # [voxel_res ** 3]

        os.makedirs(cfg.output_dir, exist_ok=True)

        if cfg.save_obj:
            print('sigma percentiles:', {q: np.percentile(sigma.reshape(-1), q) for q in [50.0, 90.0, 95.0, 97.5, 99.0, 99.5]})
            vertices, triangles = mcubes.marching_cubes(sigma, np.percentile(sigma, cfg.thresh_percentile))
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(f'{cfg.output_dir}/shape-{seed}.obj')

        if cfg.save_mrc:
            with mrcfile.new_mmap(os.path.join(cfg.output_dir, f'{seed}.mrc'), overwrite=True, shape=sigma.shape, mrc_mode=2) as mrc:
                mrc.data[:] = sigma

#----------------------------------------------------------------------------

if __name__ == "__main__":
    extract_geometry() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------