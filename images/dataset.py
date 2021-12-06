if __name__ == '__main__':
    from dataset_utils import *
    from dataset_configs import *
    from image_utils import get_location, get_vision
else:
    from .dataset_utils import *
    from .dataset_configs import *
    from .image_utils import get_location, get_vision

import pandas as pd
import numpy as np
import datetime
import glob
import os


class PandasSaver(object):
    def __init__(self, save_dir: str, save_names: tuple):
        self.save_dir = save_dir
        log_filename = datetime.datetime.now().strftime('|%Y_%m_%d|%H_%M_%S|')
        self.timed_save_dir = os.path.join(save_dir, log_filename)
        os.mkdir(self.timed_save_dir)
        for key in save_names:
            os.mkdir(os.path.join(self.timed_save_dir, key))
        self.df_dir = os.path.join(self.timed_save_dir, 'data.csv')
        self.df = pd.DataFrame.from_dict({key: [] for key in save_names})
        self.save_csv()

    def write_row(self, **kwargs):
        new_index = len(self.df.index)
        for key in kwargs:
            arr = kwargs[key]
            np.save(os.path.join(self.timed_save_dir, key, f'{new_index:09d}.npy'), arr)
            kwargs[key] = os.path.join(key, f'{new_index:09d}.npy')
        self.df = self.df.append(kwargs, ignore_index=True)

    def save_csv(self):
        self.df.to_csv(path_or_buf=self.df_dir)

    def drop_last(self, n: int):
        self.df.drop(self.df.tail(n).index, inplace=True)


class NumpyDatasetSaver(object):
    class Dataset(object):
        def __init__(self, name: str):
            self.name = name
            self.arrays = []

        def append(self, append_array: np.ndarray):
            self.arrays.append(append_array)

    def __init__(self, save_dir: str, save_names: tuple):
        for save_name in save_names:
            setattr(self, save_name, NumpyDatasetSaver.Dataset(name=save_name))
        now = datetime.datetime.now()
        log_filename = now.strftime('|%Y_%m_%d|%H_%M_%S|')
        self._save_dir = save_dir
        self._timed_save_dir = os.path.join(save_dir, log_filename)
        self._save_names = save_names

    def dump(self, ext='.npy'):

        if ext == '.npy':
            os.mkdir(self._timed_save_dir)
            for save_name in self._save_names:
                np.save(os.path.join(self._timed_save_dir, save_name + '.npy'),
                        np.stack(getattr(self, save_name).arrays, axis=0))
            self.reset()

        elif ext == '.npz':
            kwargs = dict(zip(self._save_names, (np.stack(getattr(self, save_name).arrays, axis=0)
                                                 for save_name in self._save_names)))
            np.savez_compressed(self._timed_save_dir + '.npz', **kwargs)
            self.reset()

        else:
            raise NotImplementedError(f'Data save extension type "{ext}" unimplemented.')

    def reset(self):
        self.__init__(save_dir=self._save_dir, save_names=self._save_names)


class DataLoader(object):
    class DatasetSplit(object):
        def __init__(self, data_files: pd.DataFrame, split: str, batch_size: int, root_file: str):
            self.data_files = data_files
            self.split = split
            self.batch_size = batch_size
            self.root_file = root_file
            self.seed = DATASET_SEED
            self.num_batches = np.floor(len(data_files.index) / batch_size)

        def new_seed(self):
            self.seed = np.random.RandomState(self.seed).randint(0, (2 ** 32) - 1, dtype=np.uint32)

        def prepare(self, data: dict) -> dict:
            if self.split == 'test':
                pass
            elif self.split == 'train':
                if DATASET_AUG_FLIP_LR:
                    flip_lr = np.random.RandomState(self.seed).random(data['bullet'].shape[0])
                    self.new_seed()
                    for index, flip_chance in enumerate(flip_lr):
                        if flip_chance < 0.5:
                            data['bullet'][index] = data['bullet'][index, :, ::-1]
                            data['player'][index] = data['player'][index, :, ::-1]
                            data['ctrl'][index, 3:5] = data['ctrl'][index, 4:2:-1]

                if DATASET_AUG_FLIP_UD:
                    flip_ud = np.random.RandomState(self.seed).random(data['bullet'].shape[0])
                    self.new_seed()
                    for index, flip_chance in enumerate(flip_ud):
                        if flip_chance < 0.5:
                            data['bullet'][index] = data['bullet'][index, ::-1]
                            data['player'][index] = data['player'][index, ::-1]
                            data['ctrl'][index, 1:3] = data['ctrl'][index, 2:0:-1]
            else:
                raise TypeError(f'Unknown split type "{self.split}".')

            data['bullet'] = np.expand_dims(data['bullet'].astype(np.float32) / 255, axis=-1)
            data['player'] = np.expand_dims(data['player'].astype(np.float32) / 255, axis=-1)
            data['ctrl'] = data['ctrl'].astype(np.float32)

            if DATASET_GET_VISION:
                data['player_loc'] = np.stack([get_location(arr) for arr in data['player']], axis=0)
                data['vision'] = np.stack([get_vision(loc, DATASET_VISION, arr) for loc, arr in
                                           zip(data['player_loc'], data['bullet'])])
                data['player_loc'][..., 0] /= data['player'].shape[2]
                data['player_loc'][..., 1] /= data['player'].shape[1]

            # keys: 'player', 'bullet', 'ctrl', 'player_loc', 'vision'
            return data

        def __next__(self):
            if self.batch_count < self.num_batches:

                data = self.data_files[self.batch_count*self.batch_size:(self.batch_count + 1)*self.batch_size]
                data = data.to_dict('list')
                data = {key: np.stack(list(map((
                    lambda s: np.load(os.path.join(self.root_file, s))), data[key])), axis=0) for key in data}

                data = self.prepare(data)

                self.batch_count += 1
                return data
            else:
                raise StopIteration

        def __iter__(self):
            if self.split == 'train':
                self.data_files = self.data_files.sample(frac=1.0, ignore_index=True, random_state=self.seed)
                self.new_seed()
            self.batch_cycle_end = None
            self.loaded_data = None
            self.batch_count = 0
            return self

        def __len__(self):
            return int(self.num_batches)

    def __init__(self, directory=None):
        if directory is None:
            self.root_file = glob.glob(os.path.join(DATASET_DIR, '*'))[-1]
        else:
            self.root_file = os.path.join(DATASET_DIR, directory)

        csv_df = pd.read_csv(os.path.join(self.root_file, 'data.csv'), index_col=0)

        roll = np.random.RandomState(DATASET_SEED).randint(1, len(csv_df.index))
        roll_df(csv_df, roll)

        test_len = int(np.ceil(len(csv_df.index) * DATASET_TTS))

        self.test_ds = DataLoader.DatasetSplit(
            data_files=csv_df[:test_len],
            split='test',
            batch_size=DATASET_BATCH_SIZE,
            root_file=self.root_file)
        self.train_ds = DataLoader.DatasetSplit(
            data_files=csv_df[test_len:],
            split='train',
            batch_size=DATASET_BATCH_SIZE,
            root_file=self.root_file)


class DataLoaderV1(object):
    class DatasetSplit(object):
        def __init__(self, data_files: list, split: str, batch_size: int):
            self.data_files = data_files
            self.split = split
            self.batch_size = batch_size
            self.seed = DATASET_SEED

            self.num_batches = verify_integrity(self.data_files)

        def new_seed(self):
            self.seed = np.random.RandomState(self.seed).randint(0, (2 ** 32) - 1, dtype=np.uint32)

        def prepare(self, data: dict) -> dict:
            if self.split == 'test':
                pass
            elif self.split == 'train':
                if DATASET_AUG_FLIP_LR:
                    flip_lr = np.random.RandomState(self.seed).random(data['bullet'].shape[0])
                    self.new_seed()
                    for index, flip_chance in enumerate(flip_lr):
                        if flip_chance < 0.5:
                            data['bullet'][index] = data['bullet'][index, :, ::-1]
                            data['player'][index] = data['player'][index, :, ::-1]
                            data['ctrl'][index, 3:5] = data['ctrl'][index, 4:2:-1]

                if DATASET_AUG_FLIP_UD:
                    flip_ud = np.random.RandomState(self.seed).random(data['bullet'].shape[0])
                    self.new_seed()
                    for index, flip_chance in enumerate(flip_ud):
                        if flip_chance < 0.5:
                            data['bullet'][index] = data['bullet'][index, ::-1]
                            data['player'][index] = data['player'][index, ::-1]
                            data['ctrl'][index, 1:3] = data['ctrl'][index, 2:0:-1]
            else:
                raise TypeError(f'Unknown split type "{self.split}".')

            data['bullet'] = np.expand_dims(data['bullet'].astype(np.float32) / 255, axis=-1)
            data['player'] = np.expand_dims(data['player'].astype(np.float32) / 255, axis=-1)
            data['ctrl'] = data['ctrl'].astype(np.float32)

            if DATASET_GET_VISION:
                data['player_loc'] = np.stack([get_location(arr) for arr in data['player']], axis=0)
                data['vision'] = np.stack([get_vision(loc, DATASET_VISION, arr) for loc, arr in
                                           zip(data['player_loc'], data['bullet'])])
                data['player_loc'][..., 0] /= data['player'].shape[2]
                data['player_loc'][..., 1] /= data['player'].shape[1]

            # keys: 'player', 'bullet', 'ctrl', 'player_loc', 'vision'
            return data

        def __next__(self):
            if self.batch_count < self.num_batches:

                if self.loaded_data is None:
                    file = self.data_files.pop(0)
                    self.data_files.append(file)
                    self.loaded_data = load_file(file, shuffle_seed=self.seed)
                    self.new_seed()
                    self.batch_cycle = 0
                    self.batch_cycle_end = next(iter(self.loaded_data.values())).shape[0]  # Get total data samples

                data = {key: self.loaded_data[key][self.batch_cycle] for key in self.loaded_data}
                self.batch_cycle += 1

                if self.batch_cycle == self.batch_cycle_end:  # Make sure that new data is loaded later
                    self.loaded_data = None

                data = self.prepare(data)

                self.batch_count += 1
                return data
            else:
                raise StopIteration

        def __iter__(self):
            if self.split == 'train':
                np.random.RandomState(self.seed).shuffle(self.data_files)
                self.new_seed()
            self.batch_cycle_end = None
            self.loaded_data = None
            self.batch_count = 0
            return self

        def __len__(self):
            return self.num_batches

    def __init__(self):
        data_files = glob.glob(os.path.join(DATASET_DIR, '*'))
        np.random.RandomState(DATASET_SEED).shuffle(data_files)
        test_len = int(np.ceil(len(data_files) * DATASET_TTS))

        self.test_ds = DataLoader.DatasetSplit(
            data_files=data_files[:test_len],
            split='test',
            batch_size=1,
            root_file='')
        self.train_ds = DataLoader.DatasetSplit(
            data_files=data_files[test_len:],
            split='train',
            batch_size=DATASET_BATCH_SIZE,
            root_file='')


if __name__ == '__main__':
    pass
