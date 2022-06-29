"""
This script prepares a NeRF synthetic dataset
First, we put images into a single dir

Then we extract angles from
and then creates a `dataset.json` file with these angles
"""

import os
import json
import argparse
from typing import Dict
import numpy as np

from scripts.data_scripts.resize_dataset import resize_dataset
from scripts.utils import compress_to_zip
from src.training.rendering import get_euler_angles

#----------------------------------------------------------------------------

def read_transforms_from_file(file_path: os.PathLike) -> Dict:
    with open(file_path, 'r') as f:
        transforms = json.load(f)
        transforms = {x['file_path']: x['transform_matrix'] for x in transforms['frames']}

    return transforms

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, help='Path to the NeRF scene directory')
    parser.add_argument('-t', '--target_dir', type=str, help='Where to save the result?')
    parser.add_argument('-s', '--image_size', default=256, type=int, help='Which image size to use?')
    parser.add_argument('-f', '--image_format', default='.jpg', type=str, help='Which image format to use?')
    parser.add_argument('--use_roll_angles', action='store_true', help='Should we estimate the roll angle as well?')
    parser.add_argument('--zip', action='store_true', help='Whould we put the result into a zip archive?')
    parser.add_argument('--combined_dataset', action='store_true', help='Is the dataset compbined from several collections?')
    parser.add_argument('--num_jobs', default=8, type=int, help='Number of parallel jobs when resizing the dataset.')
    args = parser.parse_args()

    resize_dataset(
        source_dir=args.directory,
        target_dir=args.target_dir,
        size=args.image_size,
        format=args.image_format,
        ignore_regex=r'.*_(normal|depth)_.*',
        num_jobs=args.num_jobs,
        images_only=True,
    )

    if args.combined_dataset:
        metadata_files = [os.path.join(args.directory, d, 'metadata.json') for d in sorted(os.listdir(args.directory))]
    else:
        metadata_files = [os.path.join(args.directory, 'metadata.json')]

    transforms = {}
    for mfile in metadata_files:
        with open(mfile, 'r') as f:
            curr_collection_name = os.path.basename(os.path.dirname(mfile)) if args.combined_dataset else ""
            curr_transforms = json.load(f)
            curr_transforms = {os.path.join(curr_collection_name, model_name, os.path.basename(t['file_path'])): t['transform_matrix'] for model_name in curr_transforms for t in curr_transforms[model_name]}
            transforms = {**curr_transforms, **transforms}

    camera_angles = {f'{f}{args.image_format}': get_euler_angles(np.array(t)) for f, t in transforms.items()}
    if not args.use_roll_angles:
        angles_values = np.array([v for v in camera_angles.values()])
        assert abs(angles_values[:, [2]]).mean() < 1e-5, f"The dataset contains roll angles: {abs(angles_values[:, 2]).sum()}."
        assert (angles_values[:, [0]] ** 2).sum() ** 0.5 > 0.1, "Broken yaw angles (all zeros)."
        assert (angles_values[:, [1]] ** 2).sum() ** 0.5 > 0.1, "Broken pitch angles (all zeros)."
        assert angles_values[:, [0]].min() >= -np.pi, f"Broken yaw angles (too small): {angles_values[:, [0]].min()}"
        assert angles_values[:, [0]].max() <= np.pi, f"Broken yaw angles (too large): {angles_values[:, [0]].max()}"
        assert angles_values[:, [1]].min() >= 0.0, f"Broken pitch angles (too small): {angles_values[:, [1]].min()}"
        assert angles_values[:, [1]].max() <= np.pi, f"Broken pitch angles (too large): {angles_values[:, [1]].max()}"

    origins = np.array([np.array(t)[:3, 3] for t in transforms.values()])
    distances = np.sqrt((origins ** 2).sum(axis=1))
    print(f'Mean/std for camera distance: {distances.mean()} {distances.std()}')
    # Saving the camera poses
    os.makedirs(args.target_dir, exist_ok=True)
    with open(os.path.join(args.target_dir, 'dataset.json'), 'w') as f:
        json.dump({'camera_angles': camera_angles}, f)

    # Creating a zip archive
    if args.zip:
        print('Creating a zip archive...')
        compress_to_zip(args.target_dir, delete=False)

#----------------------------------------------------------------------------
