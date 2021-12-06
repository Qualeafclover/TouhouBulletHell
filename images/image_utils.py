import scipy.ndimage
import typing
import itertools
import numpy as np
import cv2


def get_location(img: np.ndarray) -> typing.Tuple[float, float]:
    r"""
    :param img: 2d or 3d numpy array with 1 channel
    :return : float axis coordinates, (x, y)
    """
    out = scipy.ndimage.measurements.center_of_mass(img)
    if len(out) == 3:
        return out[:-1][::-1]
    else:
        return out[::-1]


def pol2cart(rho: float, phi: float, origin=(0., 0.)) -> typing.Tuple[float, float]:
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x+origin[0], y+origin[1]


def get_vision(pos: typing.Union[typing.List[float], np.ndarray], angles: int, img: np.ndarray) -> np.ndarray:
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
        raise f'Unknown img array shape, "{shape}".'
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
         img[arr[:, 1].astype(np.int), arr[:, 0].astype(np.int)]
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
                    start_point: typing.Union[typing.List[float], np.ndarray],
                    relative_start_point: bool,
                    length: float, radian: float,
                    color: typing.Union[typing.Tuple[int, int, int], int, float],
                    thickness: float) -> np.ndarray:
    if len(image.shape) == 2:
        h, w = image.shape
    elif len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        raise f'Unknown img array shape, "{image.shape}".'

    if relative_start_point:
        start_point = start_point[0] * w, start_point[1] * h

    max_len = (h ** 2 + w ** 2) ** 0.5

    end_point = list(map(int, pol2cart(length * max_len, radian, start_point)))
    start_point = list(map(int, start_point))
    return cv2.line(image, start_point, end_point, color, thickness)


def draw_polar_lines(image: np.ndarray,
                     start_point: typing.Union[typing.List[float], np.ndarray],
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
