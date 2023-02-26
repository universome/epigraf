"""
Adapted from https://github.com/universome/alis/blob/master/download_lhq.py
"""
import os
import math
import argparse
import requests
import shutil
import urllib
from urllib import parse as urlparse

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """
    Copy-pasted from https://stackoverflow.com/a/53877507/2685677
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file_with_progress(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


DATASET_TO_LINKS = {
    'plants': [
        'https://disk.yandex.ru/d/LJTtzCp82CE4MA',
    ],
    'food': [
        'https://disk.yandex.ru/d/yo6Mxi_zK5RqRQ',
    ],
    'plants-raw': [
        'https://disk.yandex.ru/d/OdSHLGWeDGV1yw',
        'https://disk.yandex.ru/d/1sHxMzdzEg_fxg',
        'https://disk.yandex.ru/d/aHgy8WpdnrXn7w',
        'https://disk.yandex.ru/d/sAA3AqlZAJheXQ',
        'https://disk.yandex.ru/d/nLCdYaZ5waIGrg',
    ],
    'food-raw': [
        'https://disk.yandex.ru/d/oiFYT4gslpVTew',
        'https://disk.yandex.ru/d/umyuZSmIAJfDlg',
    ],
}

API_ENDPOINT = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}'


def get_real_direct_link(sharing_link: str) -> str:
    pk_request = requests.get(API_ENDPOINT.format(sharing_link))

    return pk_request.json()['href']


def download_file(url: str, save_path: str):
    local_filename = url.split('/')[-1]

    with requests.get(url, stream=True) as r:
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def convert_size(size_bytes: int):
    """
    Copy-pasted https://stackoverflow.com/a/14822210/2685677
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return "%s %s" % (s, size_name[i])


def download_dataset(dataset_name: str, save_dir: os.PathLike):
    assert dataset_name in list(DATASET_TO_LINKS.keys()), \
        f"Wrong dataset name. Possible options are: {', '.join(DATASET_TO_LINKS.keys())}"
    file_urls = DATASET_TO_LINKS[dataset_name]

    print(f'Saving files into {save_dir} directory...')
    os.makedirs(save_dir, exist_ok=True)

    for i, file_url in enumerate(file_urls):
        download_url = get_real_direct_link(file_url)
        url_parameters = urlparse.parse_qs(urlparse.urlparse(download_url).query)
        filename = url_parameters['filename'][0]
        file_size = convert_size(int(url_parameters['fsize'][0]))
        save_path = os.path.join(save_dir, filename)
        print(f'Downloading {i+1}/{len(file_urls)} files: {save_path} (size: {file_size})')
        download_file_with_progress(download_url, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Megascans dataset downloader")
    parser.add_argument('dataset', type=str, choices=list(DATASET_TO_LINKS.keys()))
    parser.add_argument('save_dir', type=str, default='./megascans-data')
    args = parser.parse_args()

    download_dataset(
        dataset_name=args.dataset,
        save_dir=args.save_dir,
    )
