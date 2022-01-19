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
    batch_size: int = DATASET_BATCH_SIZE
    data_style: str = DATASET_DATA_STYLE
    stacks: int = DATASET_STACKS
    stack_frame_skip: int = DATASET_STACK_FRAME_SKIP
    stack_drop: bool = DATASET_STACK_DROP  # ??
    smoothen: int = DATASET_SMOOTHEN


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


class MemoryGlobber(object):
    def __init__(self):
        self.previous_request = None
        self.previous_output = None

    def glob(self, *args, **kwargs):
        request = args, kwargs
        if request != self.previous_request:
            self.previous_request = args, kwargs
            self.previous_output = glob.glob(*args, **kwargs)
        return self.previous_output


def pol2cart(rho: float, phi: float, origin=(0., 0.)) -> typing.Tuple[float, float]:
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x+origin[0], y+origin[1]


def get_vision(pos: typing.Union[tuple, list], angles: int, img: np.ndarray) -> np.ndarray:
    r"""
    :param pos: float axis coordinates, (x, y)
    :param angles: number of lines to draw
    :param img: 2d numpy array
    :return: array in format [[angles in radians], [distances], [1 for bullet, 0 for wall hit]]
    """
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

    check_lines = map(
        (lambda arr:
         np.delete(arr, np.where(
             (w <= arr[:, 0]) | (arr[:, 0] <= 0) |
             (h <= arr[:, 1]) | (arr[:, 1] <= 0)
         )[0], axis=0)),
        check_lines)  # x, y all points to check, excluding out of bounds

    lines = list(map(
        (lambda arr:
         img[arr[:, 1].astype(int), arr[:, 0].astype(int)]
         ),
        check_lines))  # values of images on check_line positions

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
            self._memory_glob = MemoryGlobber()

            data = (glob.glob(os.path.join(data_file, '*.json')) for data_file in data_files)
            data = [item for sublist in data for item in sublist]

            if self.configs.stack_drop:
                data = list(filter(self.stack_drop, data))
            if self.split == 'test':
                self.configs.batch_size = 1

            if self.configs.preload_level >= 1:  # Cluster files
                self.pb = ProgressBar(total=len(data), prefix=f'Preload 1/{self.configs.preload_level} (to stack)')
                data = list(map(self.path2stack, data, itertools.repeat(True)))
            if self.configs.preload_level >= 2:  # Load to json
                self.pb = ProgressBar(total=len(data), prefix=f'Preload 2/{self.configs.preload_level} (to json)')
                data = list(map(self.stack2json, data, itertools.repeat(True)))
            if self.configs.preload_level >= 3:  # Load to array
                self.pb = ProgressBar(total=len(data), prefix=f'Preload 3/{self.configs.preload_level} (to array)')
                data = list(map(self.json2array, data, itertools.repeat(True)))
            if self.configs.preload_level >= 4:  # Load to vision array
                self.pb = ProgressBar(total=len(data), prefix=f'Preload 4/{self.configs.preload_level} (to vision)')
                data = list(map(self.array2vision, data, itertools.repeat(True)))

            self.data = data
            self.num_batches = np.floor(len(self.data) / self.configs.batch_size)
            self.batch_count = None

        def __iter__(self):
            if self.split == 'train':
                self.random().shuffle(self.data)
            self.batch_count = 0
            return self

        def __next__(self):
            if self.batch_count < self.num_batches:
                data = self.data[self.batch_count*self.configs.batch_size:(self.batch_count+1)*self.configs.batch_size]

                if self.configs.preload_level < 1:
                    data = list(map(self.path2stack, data))
                if self.configs.preload_level < 2:
                    data = list(map(self.stack2json, data))
                if self.configs.preload_level < 3:
                    data = list(map(self.json2array, data))
                if self.configs.preload_level < 4:
                    data = list(map(self.array2vision, data))

                data = self.augmentation(data)

                # Change data from List of Dictionaries to Dictionary of Lists
                data = {key: np.stack([d[key] for d in data], axis=0) for key in data[0]}
                data['key'] = list(map(self.key_dict2arr, data['key']))
                data['key'] = np.stack(data['key'])
                data = self.filter_data(data)

                self.batch_count += 1
                return data
            else:
                raise StopIteration

        def __len__(self):
            return int(self.num_batches)

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

        @classmethod
        def filter_data(cls, data: dict) -> dict:
            new_data = {
                'hit_vision': data['hit_vision'][..., 1:3],
                'pos': np.stack([data['pos'][..., 1] / 200, (data['pos'][..., 2] - 225) / 450], axis=1),
                'key': data['key']
            }
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
            angles = data['hit_vision'].shape[0]
            data['hit_vision'][:, 1:3] = np.roll(np.roll(data['hit_vision'][:, 1:3],
                                                 -int(round(angles/4)), axis=0)[::-1],
                                                 int(round(angles/4)) + 1, axis=0)
            data['key']['left'], data['key']['right'] = data['key']['right'], data['key']['left']
            data['pos'][1] = -data['pos'][1]
            return data

        @classmethod
        def aug_flip_ud(cls, data: dict) -> dict:
            data['hit_vision'][:, 1:3] = np.roll(data['hit_vision'][::-1, 1:3], 1, axis=0)
            data['key']['up'], data['key']['down'] = data['key']['down'], data['key']['up']
            data['pos'][2] = 450 - data['pos'][2]
            return data

        def stack_drop(self, path: str):
            self_number = int(os.path.splitext(os.path.split(path)[1])[0])
            threshold = self.configs.stacks * self.configs.stack_frame_skip
            return self_number > threshold

        def path2stack(self, path: str, verbose: bool = False) -> list:
            candidates = self._memory_glob.glob(f'{os.path.split(path)[0]}/*.json')
            candidate_nums = np.array(list(map((lambda s: int(os.path.splitext(os.path.split(s)[1])[0])), candidates)))
            self_number = int(os.path.splitext(os.path.split(path)[1])[0])
            target_numbers = range(self_number - (self.configs.stacks * self.configs.stack_frame_skip), self_number,
                                   self.configs.stack_frame_skip)
            selected_index = map((lambda n: np.argmin(np.abs(candidate_nums - n))), target_numbers)
            stacks = list(map((lambda n: candidates[n]), selected_index))
            if verbose:
                self.pb.progress()
                self.pb.print(path)
            return stacks

        def stack2json(self, stack: list, verbose: bool = False) -> list:
            def load_json(path):
                with open(path, 'r') as f:
                    return json.load(f)

            json_data = list(map(load_json, stack))
            if verbose:
                self.pb.progress()
                self.pb.print(stack[0])
            return json_data

        def json2array(self, json_data_list: list, verbose: bool = False) -> list:
            def to_array(json_data):
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

                return {'hit_array': hit_array, 'key': key, 'pos': pos}
            array_data = list(map(to_array, json_data_list))
            if verbose:
                self.pb.progress()
                self.pb.print()
            return array_data

        def array2vision(self, array_data_list: list, verbose: bool = False) -> list:
            def to_vision(array_data: dict):
                pos = array_data['pos'][1] + 200, array_data['pos'][2]
                hit_vision = get_vision(pos=pos, angles=self.configs.angles, img=array_data['hit_array'])
                return {
                    'hit_vision': hit_vision,
                    'key': array_data['key'], 'pos': array_data['pos']
                }
            vision_data = list(map(to_vision, array_data_list))
            if verbose:
                # canvas = draw_polar_lines(array_data['hit_array'], pos, False, hit_vision, 150, 1)
                self.pb.progress()
                self.pb.print()
            return vision_data

        def random(self):
            random_state = np.random.RandomState(self.configs.seed)
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
    # dl = DataLoader(path='C:/Users/quale/Desktop/TouhouBulletHell/json_dataset',
    dl = DataLoader()

    for data_ in dl.train_ds:
        print(data_)
        break
