import os
import json
import argparse
from typing import List, Union
from tqdm import tqdm


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def clean_directory(directory: os.PathLike, max_iter: Union[int, str], verbose: bool=False, print_only: bool=False, ckpt_select_metric: str='fid50k_full', **kwargs):
    assert not max_iter is None, f'Please, specify `max_iter`'

    num_checkpoints_removed = 0
    num_permission_errors = 0
    amount_of_space_freed = 0
    checkpoint_dirs = collect_directories(directory, verbose=verbose, **kwargs)

    for checkpoints_dir_path in tqdm(checkpoint_dirs):
        assert os.path.isdir(checkpoints_dir_path), f"Not a directory: {checkpoints_dir_path}"

        checkpoints = [f for f in os.listdir(checkpoints_dir_path) if f.startswith('network-snapshot') and f.endswith('.pkl')]
        checkpoints = sorted(checkpoints)

        if max_iter == 'best':
            metrics_file = os.path.join(checkpoints_dir_path, f'metric-{ckpt_select_metric}.jsonl')
            if not os.path.isfile(metrics_file):
                print(f'Skipping {checkpoints_dir_path} since {ckpt_select_metric} metric was not found.')
                continue
            with open(metrics_file, 'r') as f:
                snapshot_metrics_vals = [json.loads(line) for line in f.read().splitlines()]
            best_snapshot = sorted(snapshot_metrics_vals, key=lambda m: m['results'][ckpt_select_metric])[0]
            checkpoint_to_keep = best_snapshot['snapshot_pkl']
            checkpoints_to_delete = [c for c in checkpoints if c != checkpoint_to_keep]
            # print(f'Best checkpoint: {checkpoint_to_keep}')
            # print(f'Deleting checkpoints: {checkpoints_to_delete}')
        elif max_iter == -1:
            checkpoints_to_delete = checkpoints[:-1]
        else:
            suffix_len = len('.pkl')
            iters = [int(c[c.rfind('-') + 1:-suffix_len]) for c in checkpoints]
            checkpoints_to_delete = [c for c, i in zip(checkpoints, iters) if i <= max_iter]

        if len(checkpoints_to_delete) == 0:
            if verbose:
                print(f'Didnt find any checkpoints for {checkpoints_dir_path}')
            continue

        if verbose and max_iter is None:
            print(f'Keeping only {checkpoints[-1]} ckpt inside {checkpoints_dir_path} experiment.')

        for ckpt in checkpoints_to_delete:
            ckpt_path = os.path.join(checkpoints_dir_path, ckpt)
            ckpt_size = os.path.getsize(ckpt_path)

            try:
                if print_only:
                    print(f'Pseudo-removing {ckpt_path}')
                else:
                    os.remove(ckpt_path)
            except PermissionError:
                num_permission_errors += 1
                continue

            num_checkpoints_removed += 1
            amount_of_space_freed += ckpt_size

    print(f'Deleted {num_checkpoints_removed} checkpoints. Couldnt remove {num_permission_errors} checkpoints due to PermissionError.')
    print(f'Freed up {sizeof_fmt(amount_of_space_freed)} space.')


def collect_directories(root_dir: os.PathLike, recursive: bool=False, verbose: bool=False, ignore_dir_names: List[str]=[], check_root_dir: bool=False) -> List[os.PathLike]:
    dirs_list = []
    dirs_to_check = os.listdir(root_dir) + ([root_dir] if check_root_dir else [])

    for exp_name in sorted(dirs_to_check):
        exp_path = os.path.join(root_dir, exp_name)

        if not os.path.isdir(exp_path) or os.path.islink(exp_path) or os.path.basename(exp_path) in ignore_dir_names:
            continue
        else:
            if recursive:
                dirs_list.extend(collect_directories(exp_path, recursive=recursive, verbose=verbose, ignore_dir_names=ignore_dir_names))

        checkpoints_dir_path = os.path.join(exp_path, 'output')

        if not os.path.isdir(checkpoints_dir_path) or os.path.islink(checkpoints_dir_path):
            if verbose: print(f'Didnt find the checkpoints dir for {exp_name}')
            continue

        dirs_list.append(checkpoints_dir_path)

    return dirs_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--max_iter', required=True, help='\
        Do not delete checkpoints which are older than `max_iter` iterations. \
        Specify `-1` to delete all the checkpoints except for the last one. \
        Specify best to delete all the checkpoints except for the best one acc (according to the `ckpt_select_metric` metric).')
    parser.add_argument('--verbose', action='store_true', help='Should we print intermediate progress information?')
    parser.add_argument('--recursive', action='store_true', help='Should we walk over the directories recursively?')
    parser.add_argument('--ignore_dir_names', type=str, help='Ignore directories with one of these names.')
    parser.add_argument('--print_only', action='store_true', help='Print only and exit or really remove?')
    parser.add_argument('--single_dir', action='store_true', help='Should we check only a single experiment directory?')
    parser.add_argument('--ckpt_select_metric', type=str, default='fid50k_full', help='Checkpoint selection metric for max_iter=best')
    args = parser.parse_args()

    ignore_dir_names = [] if args.ignore_dir_names is None else args.ignore_dir_names.split(',')

    clean_directory(
        directory=args.directory,
        verbose=args.verbose,
        recursive=args.recursive,
        ignore_dir_names=ignore_dir_names,
        max_iter=args.max_iter,
        print_only=args.print_only,
        check_root_dir=args.single_dir,
        ckpt_select_metric=args.ckpt_select_metric,
    )
