import os
import cv2
import glob
import json
import time
import typing
import datetime
import itertools
import numpy as np


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
        def __init__(self, data_files: typing.Iterable[str], split: str, preload_level: int, seed: int, angles: int,
                     batch_size: int):
            self.split = split
            self.seed = seed
            self.angles = angles
            self.batch_size = batch_size
            self.preload_level = preload_level
            data = (glob.glob(os.path.join(data_file, '*.json')) for data_file in data_files)
            data = [item for sublist in data for item in sublist]
            if self.preload_level >= 1:  # Load to json
                self.pb = ProgressBar(total=len(data), prefix=f'Preload 1/{self.preload_level} (to json)')
                data = list(map(self.path2json, data, itertools.repeat(True)))
            if self.preload_level >= 2:  # Load to array
                self.pb = ProgressBar(total=len(data), prefix=f'Preload 2/{self.preload_level} (to array)')
                data = list(map(self.json2array, data, itertools.repeat(True)))
            if self.preload_level >= 3:  # Load to vision array
                self.pb = ProgressBar(total=len(data), prefix=f'Preload 3/{self.preload_level} (to vision)')
                data = list(map(self.array2vision, data, itertools.repeat(True)))
            self.data = data
            self.num_batches = np.floor(len(self.data) / self.batch_size)
            self.batch_count = None

        def __iter__(self):
            if self.split == 'train':
                self.random().shuffle(self.data)
            self.batch_count = 0
            return self

        def __next__(self):
            if self.batch_count < self.num_batches:
                data = self.data[self.batch_count*self.batch_size:(self.batch_count+1)*self.batch_size]

                if self.preload_level < 1:
                    data = list(map(self.path2json, data))
                if self.preload_level < 2:
                    data = list(map(self.json2array, data))
                if self.preload_level < 3:
                    data = list(map(self.array2vision, data))

                data = self.augmentation(data)

                self.batch_count += 1
                return data
            else:
                raise StopIteration

        def __len__(self):
            return int(self.num_batches)

        def filter_data(self):
            pass

        def augmentation(self, data: list):
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
            hit_vision = get_vision(pos=pos, angles=self.angles, img=array_data['hit_array'])
            vision_data = {
                'hit_vision': hit_vision,
                'key': array_data['key'], 'pos': array_data['pos']
            }
            if verbose:
                # canvas = draw_polar_lines(array_data['hit_array'], pos, False, hit_vision, 150, 1)
                self.pb.progress()
                self.pb.print()
            return vision_data

        def random(self):
            random_state = np.random.RandomState(self.seed)
            self.seed = random_state.randint(0, (2 ** 32) - 1, dtype=np.uint32)
            return random_state

    def __init__(self, path: str, train_test_split: float, seed: int, preload_level: int, angles: int,
                 batch_size: int):
        glob_files = glob.glob(os.path.join(path, '*'))
        np.random.RandomState(seed).shuffle(glob_files)
        test_len = int(np.ceil(len(glob_files) * train_test_split))
        self.test_ds = DataLoader.DataSplit(data_files=glob_files[:test_len],
                                            split='test', preload_level=preload_level, seed=seed,
                                            angles=angles, batch_size=1)
        self.train_ds = DataLoader.DataSplit(data_files=glob_files[test_len:],
                                             split='train', preload_level=preload_level, seed=seed,
                                             angles=angles, batch_size=batch_size)


if __name__ == '__main__':
    # dl = DataLoader(path='C:/Users/quale/Desktop/TouhouBulletHell/json_dataset',
    dl = DataLoader(path='/home/shin/Desktop/TouhouBulletHell/json_dataset',
                    train_test_split=0.2, seed=42,
                    preload_level=0, angles=256, batch_size=1)

    for data_ in dl.train_ds:
        print(data_)
        break
