
import hashlib
import os
import os.path as osp
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

import requests

try:
    from tqdm import tqdm
except:

    class tqdm:
        def __init__(self, total=None):
            self.total = total
            self.n = 0

        def update(self, n):
            self.n += n
            if self.total is None:
                sys.stderr.write(f"\r{self.n:.1f} bytes")
            else:
                sys.stderr.write(
                    "\r{:.1f}%".format(100 * self.n / float(self.total))
                )
            sys.stderr.flush()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stderr.write('\n')


import logging

logger = logging.getLogger(__name__)



WEIGHTS_HOME = osp.expanduser("~/.cache/weights")

DOWNLOAD_RETRY_LIMIT = 3


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') or path.startswith('https://')



def _get_download(url, fullname):
    # using requests.get method
    fname = osp.basename(fullname)
    try:
        req = requests.get(url, stream=True)
    except Exception as e:  # requests.exceptions.ConnectionError
        logger.info(
            "Downloading {} from {} failed with exception {}".format(
                fname, url, str(e)
            )
        )
        return False

    if req.status_code != 200:
        raise RuntimeError(
            "Downloading from {} failed with code "
            "{}!".format(url, req.status_code)
        )

    # For protecting download interupted, download to
    # tmp_fullname firstly, move tmp_fullname to fullname
    # after download finished
    tmp_fullname = fullname + "_tmp"
    total_size = req.headers.get('content-length')
    with open(tmp_fullname, 'wb') as f:
        if total_size:
            with tqdm(total=(int(total_size) + 1023) // 1024) as pbar:
                for chunk in req.iter_content(chunk_size=1024):
                    f.write(chunk)
                    pbar.update(1)
        else:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    shutil.move(tmp_fullname, fullname)

    return fullname


def _wget_download(url, fullname):
    # using wget to download url
    tmp_fullname = fullname + "_tmp"
    # –user-agent
    command = 'wget -O {} -t {} {}'.format(
        tmp_fullname, DOWNLOAD_RETRY_LIMIT, url
    )
    subprc = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _ = subprc.communicate()

    if subprc.returncode != 0:
        raise RuntimeError(
            '{} failed. Please make sure `wget` is installed or {} exists'.format(
                command, url
            )
        )

    shutil.move(tmp_fullname, fullname)

    return fullname


_download_methods = {
    'get': _get_download,
    'wget': _wget_download,
}


def _download(url, path, md5sum=None, method='get'):
    """
    Download from url, save to path.

    url (str): download url
    path (str): download to given path
    md5sum (str): md5 sum of download package
    method (str): which download method to use. Support `wget` and `get`. Default is `get`.

    """
    assert method in _download_methods, 'make sure `{}` implemented'.format(
        method
    )

    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    logger.info(f"Downloading {fname} from {url}")
    while not (osp.exists(fullname) and _md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            print("retry_cnt: ", retry_cnt)
            raise RuntimeError(
                "Download from {} failed. " "Retry limit reached".format(url)
            )

        if not _download_methods[method](url, fullname):
            time.sleep(1)
            continue

    return fullname


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logger.info(f"File {fullname} md5 checking...")
    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logger.info(
            "File {} md5 check failed, {}(calc) != "
            "{}(base)".format(fullname, calc_md5sum, md5sum)
        )
        return False
    return True


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    logger.info(f"Decompressing {fname}...")

    # For protecting decompressing interupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # successed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.

    if tarfile.is_tarfile(fname):
        uncompressed_path = _uncompress_file_tar(fname)
    elif zipfile.is_zipfile(fname):
        uncompressed_path = _uncompress_file_zip(fname)
    else:
        raise TypeError(f"Unsupport compress file type {fname}")

    return uncompressed_path


def _uncompress_file_zip(filepath):
    with zipfile.ZipFile(filepath, 'r') as files:
        file_list = files.namelist()

        file_dir = os.path.dirname(filepath)

        if _is_a_single_file(file_list):
            rootpath = file_list[0]
            uncompressed_path = os.path.join(file_dir, rootpath)
            files.extractall(file_dir)

        elif _is_a_single_dir(file_list):
            # `strip(os.sep)` to remove `os.sep` in the tail of path
            rootpath = os.path.splitext(file_list[0].strip(os.sep))[0].split(
                os.sep
            )[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)

            files.extractall(file_dir)
        else:
            rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            if not os.path.exists(uncompressed_path):
                os.makedirs(uncompressed_path)
            files.extractall(os.path.join(file_dir, rootpath))

        return uncompressed_path


def _uncompress_file_tar(filepath, mode="r:*"):
    with tarfile.open(filepath, mode) as files:
        file_list = files.getnames()

        file_dir = os.path.dirname(filepath)

        if _is_a_single_file(file_list):
            rootpath = file_list[0]
            uncompressed_path = os.path.join(file_dir, rootpath)
            files.extractall(file_dir)
        elif _is_a_single_dir(file_list):
            rootpath = os.path.splitext(file_list[0].strip(os.sep))[0].split(
                os.sep
            )[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            files.extractall(file_dir)
        else:
            rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            if not os.path.exists(uncompressed_path):
                os.makedirs(uncompressed_path)

            files.extractall(os.path.join(file_dir, rootpath))

        return uncompressed_path


def _is_a_single_file(file_list):
    if len(file_list) == 1 and file_list[0].find(os.sep) < 0:
        return True
    return False


def _is_a_single_dir(file_list):
    new_file_list = []
    for file_path in file_list:
        if '/' in file_path:
            file_path = file_path.replace('/', os.sep)
        elif '\\' in file_path:
            file_path = file_path.replace('\\', os.sep)
        new_file_list.append(file_path)

    file_name = new_file_list[0].split(os.sep)[0]
    for i in range(1, len(new_file_list)):
        if file_name != new_file_list[i].split(os.sep)[0]:
            return False
    return True