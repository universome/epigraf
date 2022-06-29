"""
Given a path to a directory (or a zip archive),
it creates a dataset.json file for it with class labels from the directory structure
"""

import os
import json
import shutil
import argparse
from typing import Dict

from scripts.utils import extract_zip, compress_to_zip, file_ext
from src.training.dataset import remove_root


#----------------------------------------------------------------------------

def collect_labels_for_dir(data_path: os.PathLike, output_path: os.PathLike, overwrite: bool=False):
    if file_ext(data_path) == '.zip':
        input_is_zip = True
        print('Extracting the archive...', end='')
        extract_zip(data_path, overwrite=overwrite)
        print('done!')
        data_path = data_path[:-4]
    else:
        input_is_zip = False

    if os.path.isfile(os.path.join(data_path, 'dataset.json')):
        print('Found an existing dataset.json')
        with open(os.path.join(data_path, 'dataset.json'), 'r') as f:
            dataset_desc = json.load(f)
    else:
        dataset_desc = {}

    if file_ext(output_path) == '.zip':
        zip = True
        output_path = output_path[:-4]
    else:
        zip = False

    print('Collecting labels...')
    dataset_desc['labels'] = collect_class_labels(data_path, ignore_file='dataset.json')
    print('Collected labels!')

    if not input_is_zip:
        print('Copying the dir...', end='')
        shutil.copytree(data_path, output_path)
        print('done!')
    else:
        if not data_path != output_path:
            os.rename(data_path, output_path)

    print('Saving dataset.json...', end='')
    with open(os.path.join(output_path, 'dataset.json'), 'w') as f:
        json.dump(dataset_desc, f)
    print('done!')

    print('Saving the dataset...', end='')
    if zip:
        compress_to_zip(output_path, delete=False)
    print('done!')

#----------------------------------------------------------------------------

def collect_class_labels(data_dir: os.PathLike, ignore_file: str=None) -> Dict:
    """
    Collect labels given a file structure
    We assume that a separate class is a separate directory of directories
    """
    assert os.path.isdir(data_dir), f"Not a directory: {data_dir}"
    # Step 1: collect locations for each image
    all_files = {os.path.relpath(os.path.join(root, fname), start=data_dir) for root, _dirs, files in os.walk(data_dir) for fname in files if fname != ignore_file}
    file2grandparent = {f: os.path.dirname(os.path.dirname(f)) for f in all_files}
    grandparent2id = {p: i for i, p in enumerate(set(file2grandparent.values()))}
    dataset_name = os.path.basename(data_dir)
    labels = {remove_root(f, dataset_name): grandparent2id[p] for f, p in file2grandparent.items()}

    print(f'Found {len(grandparent2id)} different class labels.')

    return labels

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset (zip or directory)')
    parser.add_argument('--output_path', type=str, help='Where to save the result?')
    parser.add_argument('--overwrite', action='store_true', help='Whould we delete the existing zip archive?')
    # parser.add_argument('--zip', action='store_true', help='Should we store as zip?')
    args = parser.parse_args()

    collect_labels_for_dir(
        data_path=args.data_path,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )

#----------------------------------------------------------------------------
