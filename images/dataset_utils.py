from .dataset_configs import *

from numpy.lib import format
import pandas as pd
import numpy as np
import zipfile
import glob
import os


def npz_stats(npz_dir: str):
    with zipfile.ZipFile(npz_dir) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue
            npy = archive.open(name)
            version = np.lib.format.read_magic(npy)
            shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
            yield name[:-4], shape, dtype


def npy_stats(npy_dir: str) -> tuple:
    with open(npy_dir, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
    return shape, dtype


def data2comparable(data: dict):
    return {key: (data[key][0][1:], data[key][1]) for key in data}


def data2depth(data: dict):
    return [data[key][0][0] for key in data]


def batch_reshape(arr: np.ndarray, shuffle_seed: int = None) -> np.ndarray:
    data_count = arr.shape[0]
    batch_count = int(np.floor(data_count/DATASET_BATCH_SIZE))
    usable_data_count = batch_count * DATASET_BATCH_SIZE
    arr = arr[:usable_data_count]
    if shuffle_seed is not None:
        np.random.RandomState(shuffle_seed).shuffle(arr)
    arr = np.reshape(arr, (batch_count, DATASET_BATCH_SIZE) + arr.shape[1:])
    return arr


def verify_integrity(data_files: list) -> int:
    full_data = list(map(load_stats, data_files))
    sample_data = data2comparable(full_data[0])
    total_samples = 0
    for data in full_data:
        depths = data2depth(data)
        assert (depths.count(depths[0]) == len(depths)), f'Channel depth for "{data}" unequal.'
        assert (sample_data == data2comparable(
            data)), f'Incorrect data formatting when comparing "{full_data[0]}" to {data}'
        samples = depths[0]
        samples = np.floor(samples / DATASET_BATCH_SIZE)
        total_samples += samples
    return int(total_samples)


def load_stats(data_file: str) -> dict:
    _, ext = os.path.splitext(data_file)
    if ext == '':
        numpy_files = glob.glob(os.path.join(data_file, '*.npy'))
        numpy_files_name = list(map((lambda s: os.path.split(os.path.splitext(s)[0])[-1]), numpy_files))
        data_dict = dict(zip(numpy_files_name, map(npy_stats, numpy_files)))
        return data_dict
    elif ext == '.npz':
        data_dict = {name: (shape, dtype) for name, shape, dtype in npz_stats(data_file)}
        return data_dict


def load_file(data_file: str, shuffle_seed: int = None) -> dict:
    _, ext = os.path.splitext(data_file)
    if ext == '':
        numpy_files = glob.glob(os.path.join(data_file, '*.npy'))
        numpy_files_name = list(map((lambda s: os.path.split(os.path.splitext(s)[0])[-1]), numpy_files))
        data_dict = dict(zip(numpy_files_name, map(np.load, numpy_files)))
    elif ext == '.npz':
        data_dict = dict(np.load(data_file))
    else:
        raise NotImplementedError(f'Data file format "{ext}" unimplemented.')
    data_dict = {key: batch_reshape(data_dict[key], shuffle_seed) for key in data_dict}
    return data_dict


def roll_df(df: pd.DataFrame, roll: int):
    for col in df.columns:
        df[col] = np.roll(df[col], roll)
    return df
