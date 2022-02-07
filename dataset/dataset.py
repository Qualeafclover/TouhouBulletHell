from configs import *
import os
import cv2
import glob
import json
import time
import typing
import datetime
import itertools
import dataclasses
import numpy as np


@dataclasses.dataclass()
class DatasetConfigs(object):
    path: str = DATASET_PATH
    train_test_split: float = DATASET_TTS
    seed: int = DATASET_SEED
    preload_level: int = DATASET_PRELOAD_LEVEL
    angles: int = DATASET_ANGLES
    deep_vision_depth: int = DATASET_DEEP_VISION_DEPTH
    train_batch_size: int = DATASET_TRAIN_BATCH_SIZE
    test_batch_size: int = DATASET_TEST_BATCH_SIZE
    data_style: str = DATASET_DATA_STYLE
    stack_style: str = DATASET_STACK_STYLE
    stacks: int = DATASET_STACKS
    stack_frame_skip: int = DATASET_STACK_FRAME_SKIP
    smoothen: int = DATASET_SMOOTHEN


class DatasetIndexer(object):
    def __init__(self, sorted_data: list, configs: DatasetConfigs):
        self.configs = configs
        self.index = None

        prev_sums = 0
        for recording_num, recording in enumerate(sorted_data):
            candidate_nums = np.array(list(map((lambda s: int(os.path.splitext(os.path.split(s)[1])[0])), recording)))
            target_index = np.stack(
                [candidate_nums - (n * self.configs.stack_frame_skip)
                 for n in range(self.configs.stacks)], axis=1).reshape((len(recording) * self.configs.stacks,))
            select_index = np.array(
                list(map((lambda n: np.argmin(np.abs(candidate_nums - n))),
                         target_index))).reshape((len(recording), self.configs.stacks))
            select_index += prev_sums

            prev_sums += len(sorted_data[recording_num])
            if recording_num == 0:
                self.index = select_index
            else:
                self.index = np.concatenate([self.index, select_index])

    def random(self, new_seed: bool = True):
        random_state = np.random.RandomState(self.configs.seed)
        if new_seed:
            self.configs.seed = random_state.randint(0, (2 ** 32) - 1, dtype=np.uint32)
        return random_state

    def get_filtered(self, shuffle=True) -> np.ndarray:
        output = np.array([])
        if self.configs.stack_style == 'use_all':
            output = self.index
        if self.configs.stack_style == 'skip_dupes':
            output = np.array([v for v in self.index if len(set(v)) == len(v)])
        if self.configs.stack_style == 'skip_all':
            pass
        if shuffle:
            self.random().shuffle(output)
        return output

    def __len__(self):
        return self.index.shape[0]


class ProgressBar(object):
    def __init__(self, total: int, prefix='Progress', bar_length=64):
        self.total = total
        self.current = 0
        self.bar_length = bar_length
        self.start_time = time.perf_counter()
        self.prefix = prefix

    def progress(self, amount=1):
        self.current += amount

    def print(self, *args, **kwargs):
        time_taken = time.perf_counter() - self.start_time
        if self.current == 0:
            est_left = '?:??:??'
        else:
            est_one = time_taken/self.current
            est_left = est_one * (self.total - self.current)
            est_left = datetime.timedelta(seconds=round(est_left))

        time_taken = datetime.timedelta(seconds=round(time_taken))
        progress = self.current / self.total
        block = int(round(self.bar_length*progress))
        u_block = self.bar_length - block
        text = f"\r{self.prefix} {self.current}/{self.total} [{'='*block+'>'+'.'*u_block}] " \
               f"{progress*100:.3g}% | Time left: {est_left} | Time taken: {time_taken}"
        for arg in args:
            text += f' | {arg}'
        for kwarg in kwargs:
            text += f' | {kwarg}: {kwargs[kwarg]}'
        if progress == 1.0:
            text += '\n'
        print(f'\r{text}', end='')


def pol2cart(rho: float, phi: float, origin=(0., 0.)) -> typing.Tuple[float, float]:
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x+origin[0], y+origin[1]


def get_vision(pos: typing.Union[tuple, list], angles: int, img: np.ndarray, mode='simple_vision') -> np.ndarray:
    r"""
    :param pos: float axis coordinates, (x, y)
    :param angles: number of lines to draw
    :param img: 2d numpy array
    :param mode: vision method, must be "simple_vision" or "deep_vision"
    :return: array in format [[angles in radians], [distances], [1 for bullet, 0 for wall hit]]
    """
    assert mode in ('simple_vision', 'deep_vision')
    shape = img.shape
    if len(shape) == 2:
        h, w = shape
    elif len(shape) == 3:
        h, w, _ = shape
        img = img.squeeze(axis=-1)
    else:
        raise Exception(f'Unknown img array shape, "{shape}".')
    max_len = int(np.ceil((h ** 2 + w ** 2) ** 0.5))
    check_angles = np.linspace(0, np.pi*2, angles, endpoint=False)

    check_lines = map(pol2cart,
                      itertools.repeat(max_len), check_angles, itertools.repeat(pos))  # x, y end points

    check_lines = map(np.linspace,
                      itertools.repeat(pos), check_lines, itertools.repeat(max_len))  # x, y positions to check

    if mode == 'simple_vision':
        check_lines = map(
            (lambda arr:
             np.delete(arr, np.where(
                 (w <= arr[:, 0]) | (arr[:, 0] <= 0) |
                 (h <= arr[:, 1]) | (arr[:, 1] <= 0)
             )[0], axis=0)),
            check_lines)  # x, y all points to check, excluding out of bounds

    elif mode == 'deep_vision':  # TODO you left off here trying to make the out of bounds visible from the output array
        check_lines = map(
            (lambda arr: np.where(
                ~np.stack([(w <= arr[:, 0]) | (arr[:, 0] <= 0) | (h <= arr[:, 1]) | (arr[:, 1] <= 0)] * 2, axis=-1),
                arr, np.full_like(arr, pos)
            )),
            check_lines)  # x, y all points to check, replacing out of bounds with None

    lines = list(map(
        (lambda arr:
         img[arr[:, 1].astype(int), arr[:, 0].astype(int)]
         ),
        check_lines))  # values of images on check_line positions

    if mode == 'deep_vision':
        lines = np.stack(lines)
        print(lines.shape)
        cv2.imshow('', lines)
        cv2.waitKey(0)
        quit()

    line_switches = map(
        (lambda arr:
         np.where(arr[:-1] != arr[1:])[0]),
        lines)  # distance of angle where collision and non-collision switch

    distances = list(map(
        (lambda ls, l, ca: (ca, ls[0] / max_len, 1) if len(ls) else (ca, len(l) / max_len, 0)),
        line_switches, lines, check_angles))  # results ({angle}, {distance}, {1 for bullet, 0 for wall hit})

    distances = np.array(distances)
    return distances


def draw_polar_line(image: np.ndarray,
                    start_point: typing.Union[tuple, list],
                    relative_start_point: bool,
                    length: float, radian: float,
                    color: typing.Union[typing.Tuple[int, int, int], int, float],
                    thickness: float) -> np.ndarray:
    if len(image.shape) == 2:
        h, w = image.shape
    elif len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        raise Exception(f'Unknown img array shape, "{image.shape}".')

    if relative_start_point:
        start_point = start_point[0] * w, start_point[1] * h

    max_len = (h ** 2 + w ** 2) ** 0.5

    end_point = list(map(int, pol2cart(length * max_len, radian, start_point)))
    start_point = list(map(int, start_point))
    return cv2.line(image, start_point, end_point, color, thickness)


def draw_polar_lines(image: np.ndarray,
                     start_point: typing.Union[tuple, list],
                     relative_start_point: bool,
                     line_list: typing.Union[list, np.ndarray],
                     color: typing.Union[typing.Tuple[int, int, int], int, float],
                     thickness: float) -> np.ndarray:
    for _ in map(draw_polar_line, itertools.repeat(image),
                 itertools.repeat(start_point),
                 itertools.repeat(relative_start_point),
                 line_list[:, 1], line_list[:, 0],
                 itertools.repeat(color),
                 itertools.repeat(thickness),
                 ):
        pass  # Execute the mapping process
    return image


class DataLoader(object):
    class DataSplit(object):
        def __init__(self, data_files: typing.Iterable[str], split: str, configs: DatasetConfigs):
            self.split = split
            self.configs = configs

            self.batch_size = None
            if self.split == 'test':
                self.batch_size = self.configs.test_batch_size
            if self.split == 'train':
                self.batch_size = self.configs.train_batch_size

            data = list(sorted(glob.glob(os.path.join(data_file, '*.json'))) for data_file in data_files)
            self.indexer = DatasetIndexer(data, configs=configs)
            data = [item for sublist in data for item in sublist]

            if self.configs.preload_level >= 1:  # Load to json
                self.pb = ProgressBar(total=len(data), prefix=f'Preload 1/{self.configs.preload_level} (to json)')
                data = list(map(self.path2json, data, itertools.repeat(True)))
            if self.configs.preload_level >= 2:  # Load to array
                self.pb = ProgressBar(total=len(data), prefix=f'Preload 2/{self.configs.preload_level} (to array)')
                data = list(map(self.json2array, data, itertools.repeat(True)))
            if self.configs.preload_level >= 3:  # Load to vision array
                self.pb = ProgressBar(total=len(data), prefix=f'Preload 3/{self.configs.preload_level} (to vision)')
                data = list(map(self.array2vision, data, itertools.repeat(True)))

            self.data = data
            self.num_batches = np.floor(len(self.indexer.get_filtered(shuffle=False)) / self.batch_size)
            self.batch_count = None
            self.indexes = None

        def __iter__(self):
            if self.split == 'train':
                self.indexes = self.indexer.get_filtered(shuffle=True)
            if self.split == 'test':
                self.indexes = self.indexer.get_filtered(shuffle=True)
            self.batch_count = 0
            return self

        def __next__(self):
            if self.batch_count < self.num_batches:
                index = np.array(list(self.indexes[self.batch_count * self.batch_size + n]
                                      for n in range(self.batch_size)))
                index = index.reshape((index.shape[0] * index.shape[1],))

                data = (self.data[n] for n in index)
                if self.configs.preload_level < 1:
                    data = list(map(self.path2json, data))  # False positive warning
                if self.configs.preload_level < 2:
                    data = list(map(self.json2array, data))
                if self.configs.preload_level < 3:
                    data = list(map(self.array2vision, data))

                data = self.augmentation(data)

                # Change data from List of Dictionaries to Dictionary of Lists
                data = {key: np.stack([d[key] for d in data], axis=0) for key in data[0]}
                data['key'] = list(map(self.key_dict2arr, data['key']))
                data['key'] = np.stack(data['key'])
                data = self.restack_data(data)
                print(data['X'].shape)
                data = self.filter_data(data)

                self.batch_count += 1
                return data
            else:
                raise StopIteration

        def __len__(self):
            return int(self.num_batches)

        def restack_data(self, data: dict) -> dict:
            new_data = {
                'X': data['X'].reshape(
                    (data['X'].shape[0]//self.configs.stacks, self.configs.stacks, ) +
                    data['X'].shape[1:]),
                'pos': data['pos'].reshape(
                    (data['pos'].shape[0]//self.configs.stacks, self.configs.stacks, ) +
                    data['pos'].shape[1:]),
                'key': data['key'].reshape(
                    (data['key'].shape[0]//self.configs.stacks, self.configs.stacks, ) +
                    data['key'].shape[1:])[:, 0, :]
            }
            return new_data

        @classmethod
        def key_dict2arr(cls, key_dict: dict) -> np.ndarray:
            direction_dict = {(1, 0): 1, (0, 0): 0, (0, 1): -1, (1, 1): -1}
            movement_angle = np.arctan2(direction_dict[(key_dict['up'], key_dict['down'])],
                                        direction_dict[(key_dict['right'], key_dict['left'])], dtype=np.float32)
            movement = (1 - 0.5 * key_dict['slow']) * \
                       (key_dict['up'] or key_dict['down'] or key_dict['right'] or key_dict['left'])
            # output = np.array([np.sin(movement_angle), np.cos(movement_angle), movement], dtype=np.float32)
            output = np.array([movement * np.sin(movement_angle), movement * np.cos(movement_angle)], dtype=np.float32)
            return output

        def filter_data(self, data: dict) -> dict:
            if self.configs.data_style == 'raw':
                new_data = {
                    'X': data['X'],
                    'pos': np.stack([data['pos'][..., 1] / 200, (data['pos'][..., 2] - 225) / 450], axis=1),
                    'key': data['key']
                }
            elif self.configs.data_style == 'simple_vision':
                new_data = {
                    'X': data['X'][..., 1:3],  # Removes the radians from the X data
                    'pos': np.stack([data['pos'][..., 1] / 200, (data['pos'][..., 2] - 225) / 450], axis=1),
                    'key': data['key']
                }
            else:
                raise NotImplementedError
            return new_data

        def augmentation(self, data: list) -> list:
            if self.split == 'test':
                pass
            elif self.split == 'train':
                # Flip LR
                if self.random().random() > 0.5:
                    data = list(map(self.aug_flip_lr, data))

                # Flip UD
                if self.random().random() > 0.5:
                    data = list(map(self.aug_flip_ud, data))
            else:
                raise TypeError(f'Unknown split type "{self.split}".')
            return data

        @classmethod
        def aug_flip_lr(cls, data: dict) -> dict:
            angles = data['X'].shape[0]
            data['X'][:, 1:3] = np.roll(np.roll(data['X'][:, 1:3],
                                                -int(round(angles/4)), axis=0)[::-1],
                                        int(round(angles/4)) + 1, axis=0)
            data['key']['left'], data['key']['right'] = data['key']['right'], data['key']['left']
            data['pos'][1] = -data['pos'][1]
            return data

        @classmethod
        def aug_flip_ud(cls, data: dict) -> dict:
            data['X'][:, 1:3] = np.roll(data['X'][::-1, 1:3], 1, axis=0)
            data['key']['up'], data['key']['down'] = data['key']['down'], data['key']['up']
            data['pos'][2] = 450 - data['pos'][2]
            return data

        def path2json(self, path: str, verbose: bool = False) -> dict:
            with open(path, 'r') as f:
                json_data = json.load(f)
            if verbose:
                self.pb.progress()
                self.pb.print(path)
            return json_data

        def json2array(self, json_data: dict, verbose: bool = False) -> dict:
            enemies = json_data['enemy']
            bullets = json_data['bullet']
            key = json_data['key']
            pos = json_data['pos']

            hit_array = np.zeros(shape=(450, 400), dtype=np.uint8)
            for x, y, d, t in zip(enemies['x'], enemies['y'],
                                  enemies['diameter'], enemies['type']):
                if t != 0:
                    x, y, r = int(round(x + 200)), int(round(y)), int(np.ceil(round(d / 2)))
                    hit_array = cv2.circle(hit_array, (x, y), r, 255, -1)

            for x, y, r, rm in zip(bullets['x'], bullets['y'],
                                   bullets['radius'], bullets['radius_mult']):
                x, y, r = int(round(x + 200)), int(round(y)), int(np.ceil(round(r * rm)))
                hit_array = cv2.circle(hit_array, (x, y), r, (255, 0, 0), -1)

            array_data = {'hit_array': hit_array, 'key': key, 'pos': pos}
            if verbose:
                self.pb.progress()
                self.pb.print()
            return array_data

        def array2vision(self, array_data: dict, verbose: bool = False) -> dict:
            pos = array_data['pos'][1] + 200, array_data['pos'][2]
            if self.configs.data_style == 'raw':
                x = array_data['hit_array']
            elif self.configs.data_style == 'simple_vision':
                x = get_vision(pos=pos, angles=self.configs.angles, img=array_data['hit_array'], mode='simple_vision')
            elif self.configs.data_style == 'deep_vision':
                x = get_vision(pos=pos, angles=self.configs.angles, img=array_data['hit_array'], mode='deep_vision')
            else:
                raise TypeError
            vision_data = {
                'X': x,
                'key': array_data['key'], 'pos': array_data['pos']
            }
            if verbose:
                # canvas = draw_polar_lines(array_data['hit_array'], pos, False, X, 150, 1)
                self.pb.progress()
                self.pb.print()
            return vision_data

        def random(self, new_seed: bool = True):
            random_state = np.random.RandomState(self.configs.seed)
            if new_seed:
                self.configs.seed = random_state.randint(0, (2 ** 32) - 1, dtype=np.uint32)
            return random_state

    def __init__(self, configs=DatasetConfigs()):
        glob_files = glob.glob(os.path.join(configs.path, '*'))
        np.random.RandomState(configs.seed).shuffle(glob_files)
        test_len = int(np.ceil(len(glob_files) * configs.train_test_split))
        self.test_ds = DataLoader.DataSplit(data_files=glob_files[:test_len],
                                            split='test', configs=configs)
        self.train_ds = DataLoader.DataSplit(data_files=glob_files[test_len:],
                                             split='train', configs=configs)


if __name__ == '__main__':
    dl = DataLoader()
    for data_ in dl.test_ds:
        # cv2.imshow('', data_['X'][0, ..., 1])
        cv2.imshow('', data_['X'][0, 0])
        cv2.waitKey(1)
        print(data_['X'].shape)
        pass
