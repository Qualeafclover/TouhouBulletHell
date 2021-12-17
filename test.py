"""
from game_master import *
import time
import cv2
from pprint import pprint
import shutil
import os
import numpy as np

tm = TouhouMaster()
print(tm.pid)

directory = '/home/shin/Desktop/TouhouBulletHell/json_dataset/H10'

try:
    shutil.rmtree(directory)
except FileNotFoundError:
    pass
finally:
    os.mkdir(directory)

frame = None
while True:
    if tm.io_gamestate.read() == 'game':
        path = os.path.join(directory, '{}.json')
        tm.json(path)

# tm.io_infbomb.enable()
# tm.io_invuln.enable()
# while True:
#     if tm.io_gamestate.read() == 'game':
#         canvas = np.full((450, 400, 3), 0, dtype=np.uint8)
#
#         read = tm.enemy_list['x', 'y', 'diameter', 'type']
#         for x, y, d, t in read:
#             if t != 0:
#                 x, y, r = int(round(x + 200)), int(round(y)), int(round(d / 2))
#                 canvas = cv2.circle(canvas, (x, y), r, (255, 255, 255), -1)
#
#         read = tm.bullet_list['x', 'y', 'radius', 'radius_mult']
#         for x, y, r, rm in read:
#             x, y, r = int(round(x + 200)), int(round(y)), int(round(r * rm))
#             canvas = cv2.circle(canvas, (x, y), r, (255, 0, 0), -1)
#
#         read = tm.laser_list['x', 'y', 'direction', 'thickness', 'length', 'type']
#         for x, y, d, t, l, tp in read:
#             if tp != 0:
#                 try:
#                     x, y, r = int(round(x + 200)), int(round(y)), int(round(t / 2))
#                     x_, y_ = int(round(x + (l * np.cos(d)))), int(round(y + (l * np.sin(d))))
#                     canvas = cv2.line(canvas, (x, y), (x_, y_), (150, 150, 150), r)
#                 except Exception:
#                     print('Laser exception:', x, y, d, t, l, tp)
#
#         r, x, y = tm.io_pos.read()
#         r, x, y = int(round(r)), int(round(x + 200)), int(round(y))
#         canvas = cv2.circle(canvas, (x, y), r, (0, 0, 255), -1)
#
#         cv2.imshow('game', canvas)
#         cv2.waitKey(1)
#     else:
#         cv2.destroyAllWindows()

# tm.io_key.write(KeyState(shot=True, left=False, slow=False, right=True))

# while True:
#     # print(f'\r{tm.io_gamestate.read()}', end='    ')
#     print(f'\r{tm.io_aimove.read()}', end='     ')
#     # break
#     time.sleep(0.05)
# """

# from images import DataLoader
#
# dl = DataLoader()
#
# for _ in range(10):
#     for data in dl.train_ds:
#         print(data['ctrl'][0])
#         break

# import tensorflow as tf
#
#
# x = tf.reshape(tf.range(21, dtype=tf.float32), [7, 3])
# x_transform = tf.convert_to_tensor([-1, -0.5, 0, 0.5, 1, 1.5, 2], dtype=tf.float32)
# print(x)
# print(x_transform)
# print(tf.multiply(x, x_transform))

# import tensorflow as tf
# import numpy as np
#
# # UP LEFT
# n = np.arctan2(1, -1)
# n = np.sin(n), np.cos(n)
# print(n)
#
# n = tf.math.atan2(n[1], n[0])
# print(np.degrees(n))

import tensorflow as tf

print(tf.version)
inp = tf.keras.layers.Input(
shape=[3],
dtype=tf.dtypes.float32
)
print(inp)

t = inp[..., 1]
print(t)



