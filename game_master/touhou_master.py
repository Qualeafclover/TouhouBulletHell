from .touhou_configs import *
from pprint import pprint
from mem_edit import Process
import numpy as np
import dataclasses
import subprocess
import typing
import ctypes
import struct
import json
import time
import cv2
import os

import gi
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk

def xwininfo(win_name:str) -> dict:
    stdout = subprocess.check_output(['xwininfo', '-name', win_name])
    stdout = stdout.decode()
    stdout = stdout.split('\n  ')
    output = {
        'WindowID': int(stdout[0].split('Window id: ')[1].split(' ')[0], 16),
        'Width': int(stdout[5].split(': ')[1]),
        'Height': int(stdout[6].split(': ')[1]),
    }
    return output

def get_pid(name:str) -> int:
    return int(subprocess.check_output(['pidof', name])) # Fails if more than 1 process detected

@dataclasses.dataclass(frozen=True)
class KeyState(object):
    shot:  bool = False
    bomb:  bool = False
    slow:  bool = False
    up:    bool = False
    down:  bool = False
    left:  bool = False
    right: bool = False
    skip:  bool = False

class GameKeyIO(object):
    def __init__(self, process, address=GAME_KEYS_ADDR,
                 encoder=GAME_KEYS_ENCODER, ctype=GAME_KEYS_CTYPE):
        self.process = process
        self.address = address
        self.encoder = encoder
        self.ctype = ctype

    def read(self) -> KeyState:
        keys = self.process.read_memory(self.address, self.ctype())
        keys = str(bin(int.from_bytes(keys, 'little'))[2:])
        keys = tuple(reversed('0'*(10-len(keys)) + keys))
        keys = tuple(bool(int(keys[int(n)])) for n in self.encoder)
        return KeyState(*keys)

    def write(self, keystate:KeyState):
        keys = dataclasses.astuple(keystate)
        keylist = ['0' for _ in range(10)]
        for key, idnum in zip(keys, self.encoder):
            if key: keylist[int(idnum)] = '1'
        binary = ''.join(reversed(keylist))
        self.process.write_memory(self.address, self.ctype(int(binary, 2)))

    def __str__(self):
        return str(self.read())

class GamePosIO(object):
    def __init__(self, process,
                 edi_finder=GAME_EDI_FIND, edi_ctype=GAME_EDI_CTYPE, r_offset=GAME_EDI_R_OFFSET,
                 x_offset=GAME_EDI_X_OFFSET, y_offset=GAME_EDI_Y_OFFSET, xyr_ctype=GAME_EDI_XYR_CTYPE,
                 r_mult=GAME_R_MULT,
                 ):
        self.process = process
        self.edi_finder = edi_finder
        self.edi_ctype = edi_ctype
        self.r_offset = GAME_EDI_R_OFFSET
        self.x_offset = GAME_EDI_X_OFFSET
        self.y_offset = GAME_EDI_Y_OFFSET
        self.xyr_ctype = xyr_ctype
        self.r_mult = r_mult

    def read(self) -> tuple:
        edi = self.process.read_memory(self.edi_finder, self.edi_ctype())
        r_mult = self.process.read_memory(self.r_mult, self.xyr_ctype())

        pos_r = self.process.read_memory(int.from_bytes(edi, 'little') + self.r_offset, self.xyr_ctype())
        pos_x = self.process.read_memory(int.from_bytes(edi, 'little') + self.x_offset, self.xyr_ctype())
        pos_y = self.process.read_memory(int.from_bytes(edi, 'little') + self.y_offset, self.xyr_ctype())

        pos_r = ctypes.pointer(pos_r)
        r_mult = ctypes.pointer(r_mult)
        pos_x = ctypes.pointer(pos_x)
        pos_y = ctypes.pointer(pos_y)

        pos_r = ctypes.cast(pos_r, ctypes.POINTER(self.xyr_ctype)).contents.value
        r_mult = ctypes.cast(r_mult, ctypes.POINTER(self.xyr_ctype)).contents.value
        pos_x = ctypes.cast(pos_x, ctypes.POINTER(self.xyr_ctype)).contents.value
        pos_y = ctypes.cast(pos_y, ctypes.POINTER(self.xyr_ctype)).contents.value

        pos_r *= r_mult

        return pos_r, pos_x, pos_y

    def __str__(self):
        return str(self.read())

class GameIO(object):
    def __init__(self, process, address:hex, ctype:ctypes._SimpleCData,
                 enable:str=None, disable:str=None, translator:dict=None):
        self.process = process
        self.address = address
        self.ctype = ctype
        self.enabler = enable
        self.disabler = disable
        self.translator = translator

    def read(self):
        memory = self.process.read_memory(self.address, self.ctype())
        memory = ctypes.pointer(memory)
        memory = ctypes.cast(memory, ctypes.POINTER(self.ctype)).contents.value
        if self.translator is not None:
            memory = self.translator[memory]
        return memory

    def enable(self):
        assert self.enabler is not None
        self.process.write_memory(self.address, self.ctype(self.enabler))

    def disable(self):
        assert self.disabler is not None
        self.process.write_memory(self.address, self.ctype(self.disabler))

    def __str__(self):
        return str(self.read())

class FPS(object):
    class TimerNotStartedError(Exception): pass

    def __init__(self):
        self.started = False
        self.frames = 0

    def step(self):
        if not self.started:
            self.started = True
            self.start_time = time.perf_counter()
        self.frames += 1

    def get_fps(self, roundto=1):
        try:
            return round(self.frames/(time.perf_counter()-self.start_time))
        except:
            raise self.TimerNotStartedError

    def __str__(self):
        return str(self.get_fps())

class NestedPointers(object):
    def __init__(self, process, pointer_address:hex, list_ctype:hex, pointer_offset:hex,
                 pointer_next_offsets:hex=None, break_on:hex=0x0, lookup:dict={}, max_read=1000):
        self.process = process
        self.pointer_address = pointer_address
        self.list_ctype = list_ctype
        self.pointer_offset = pointer_offset
        self.pointer_next_offset = pointer_next_offsets
        self.break_on = break_on
        self.lookup = lookup
        self.max_read = max_read

    def read(self) -> list:
        output = []
        memory = self.process.read_memory(self.pointer_address, self.list_ctype())
        memory = int.from_bytes(memory, 'little')
        if memory == self.break_on: return output  # Game not started
        memory = self.process.read_memory(memory + self.pointer_offset, self.list_ctype())
        memory = int.from_bytes(memory, 'little')
        if memory == self.break_on: return output  # No item

        for item_num in range(self.max_read):
            output.append(memory)
            memory = self.process.read_memory(memory + self.pointer_next_offset, self.list_ctype())
            memory = int.from_bytes(memory, 'little')
            if memory == self.break_on: break
        return output

    def __getitem__(self, item:typing.Union[str, tuple]):
        locations = self.read()
        if type(item) == str:
            return list(map((
                lambda h: ctypes.cast(ctypes.pointer(
                    self.process.read_memory(h + self.lookup[item][0], self.lookup[item][1]())
                ), ctypes.POINTER(self.lookup[item][1])).contents.value
            ), locations))
        elif type(item) == tuple:
            return list(zip(*(list(map((
                lambda h: ctypes.cast(ctypes.pointer(
                    self.process.read_memory(h + self.lookup[item_][0], self.lookup[item_][1]())
                ), ctypes.POINTER(self.lookup[item_][1])).contents.value
            ), locations)) for item_ in item)))
        elif item == ...:
            return {
                item_: list(map((
                    lambda h: ctypes.cast(ctypes.pointer(
                        self.process.read_memory(h + self.lookup[item_][0], self.lookup[item_][1]())
                    ), ctypes.POINTER(self.lookup[item_][1])).contents.value
                ), locations)) for item_ in self.lookup
            }
        else:
            raise TypeError


class ScreenShot(object):
    def __init__(self):
        window = Gdk.get_default_root_window()
        self.screen = window.get_screen()

    def screenshot(self, wid:int, return_image=True, use_play_area=True, temp_storage=GAME_SCREENSHOT_TEMPFILE) -> np.ndarray:
        for window in self.screen.get_window_stack():
            if window.get_xid() == wid:
                w, h = window.get_width(), window.get_height()
                if use_play_area:
                    pb = Gdk.pixbuf_get_from_window(
                        window, GAME_WORKING_ABS_W(w), GAME_WORKING_ABS_H(h), GAME_WORKING_W(w), GAME_WORKING_H(h))
                else:
                    pb = Gdk.pixbuf_get_from_window(window, 0, 0, w, h)
                pb.savev(temp_storage, os.path.splitext(temp_storage)[1][1:], (), ())
                if return_image:
                    nparr = cv2.imread(temp_storage)
                    return nparr
                else:
                    return None

class TouhouMaster(object):
    def __init__(self):
        # Starts game if instance exists and get its PID, else it just gets its PID directly.
        try:
            self.pid = get_pid(GAME_PROCESS_NAME)
        except subprocess.CalledProcessError:
            self.start_game()
            time.sleep(8)
            self.pid = get_pid(GAME_PROCESS_NAME)

        # Get window info and screenshotter object
        self.wininfo = xwininfo(GAME_WINDOW_NAME)
        self.ss = ScreenShot()
        self.process = Process(process_id=self.pid)

        # Data IO
        self.enemy_list = NestedPointers(
            process=self.process,
            pointer_address=GAME_ENEMY_LIST_POINTER,
            list_ctype=GAME_ENEMY_LIST_CTYPE,
            pointer_offset=GAME_ENEMY_LIST_POINTER_OFFSET,
            pointer_next_offsets=GAME_ENEMY_LIST_NEXT_ITEM_OFFSET,
            break_on=GAME_ENEMY_LIST_BREAKPOINT,
            lookup=GAME_ENEMY_LIST_LOOKUP,
        )
        self.bullet_list = NestedPointers(
            process=self.process,
            pointer_address=GAME_BULLET_LIST_POINTER,
            list_ctype=GAME_BULLET_LIST_CTYPE,
            pointer_offset=GAME_BULLET_LIST_POINTER_OFFSET,
            pointer_next_offsets=GAME_BULLET_LIST_NEXT_ITEM_OFFSET,
            break_on=GAME_BULLET_LIST_BREAKPOINT,
            lookup=GAME_BULLET_LIST_LOOKUP,
        )
        self.laser_list = NestedPointers(
            process=self.process,
            pointer_address=GAME_LASER_LIST_POINTER,
            list_ctype=GAME_LASER_LIST_CTYPE,
            pointer_offset=GAME_LASER_LIST_POINTER_OFFSET,
            pointer_next_offsets=GAME_LASER_LIST_NEXT_ITEM_OFFSET,
            break_on=GAME_LASER_LIST_BREAKPOINT,
            lookup=GAME_LASER_LIST_LOOKUP,
        )

        self.io_inflife = GameIO(
            process=self.process,
            address=GAME_INFLIFE,
            ctype=GAME_INFLIFE_CTYPE,
            enable=GAME_INFLIFE_ENABLE,
            disable=GAME_INFLIFE_DISABLE,
            translator=GAME_INFLIFE_DICT,
        )
        self.io_infbomb = GameIO(
            process=self.process,
            address=GAME_INFBOMB,
            ctype=GAME_INFBOMB_CTYPE,
            enable=GAME_INFBOMB_ENABLE,
            disable=GAME_INFBOMB_DISABLE,
            translator=GAME_INFBOMB_DICT,
        )
        self.io_autobomb = GameIO(
            process=self.process,
            address=GAME_AUTOBOMB,
            ctype=GAME_AUTOBOMB_CTYPE,
            enable=GAME_AUTOBOMB_ENABLE,
            disable=GAME_AUTOBOMB_DISABLE,
            translator=GAME_AUTOBOMB_DICT,
        )
        self.io_invuln = GameIO(
            process=self.process,
            address=GAME_INVULN,
            ctype=GAME_INVLUN_CTYPE,
            enable=GAME_INVLUN_ENABLE,
            disable=GAME_INVLUN_DISABLE,
            translator=GAME_INVULN_DICT,
        )
        self.io_gamestate = GameIO(
            process=self.process,
            address=GAME_GAMESTATE,
            ctype=GAME_GAMESTATE_CTYPE,
            translator=GAME_GAMESTATE_DICT,
        )
        self.io_stage = GameIO(
            process=self.process,
            address=GAME_STAGE,
            ctype=GAME_STAGE_CTYPE,
            translator=GAME_STAGE_DICT,
        )
        self.io_aimove = GameIO(
            process=self.process,
            address=GAME_AIMOVE,
            ctype=GAME_AIMOVE_CTYPE,
            enable=GAME_AIMOVE_ENABLE,
            disable=GAME_AIMOVE_DISABLE,
            translator=GAME_AIMOVE_DICT,
        )
        self.io_power = GameIO(
            process=self.process,
            address=GAME_POWER,
            ctype=GAME_POWER_CTYPE,
        )
        self.io_score = GameIO(
            process=self.process,
            address=GAME_SCORE,
            ctype=GAME_SCORE_CTYPE,
        )
        self.io_graze = GameIO(
            process=self.process,
            address=GAME_GRAZE,
            ctype=GAME_GRAZE_CTYPE,
        )
        self.io_life = GameIO(
            process=self.process,
            address=GAME_LIFE,
            ctype=GAME_LIFE_CTYPE,
        )
        self.io_lifefrag = GameIO(
            process=self.process,
            address=GAME_LIFEFRAG,
            ctype=GAME_LIFEFRAG_CTYPE,
        )
        self.io_bomb = GameIO(
            process=self.process,
            address=GAME_BOMB,
            ctype=GAME_BOMB_CTYPE,
        )
        self.io_bombfrag = GameIO(
            process=self.process,
            address=GAME_BOMBFRAG,
            ctype=GAME_BOMBFRAG_CTYPE,
        )
        self.io_stageframe = GameIO(
            process=self.process,
            address=GAME_STAGEFRAME,
            ctype=GAME_STAGEFRAME_CTYPE,
        )

        self.io_key = GameKeyIO(process=self.process)
        self.io_pos = GamePosIO(process=self.process)

        # print(self.wininfo)
        # fps = FPS()
        # while True:
        #     image = self.screenshot(return_image=True)
        #     cv2.imshow('', image)
        #     cv2.waitKey(1)
        #     fps.step()
        #     print(f'\rfps: {fps} {self.io_stageframe}', end='   ')

    def screenshot(self, return_image=True, use_play_area=True, temp_storage=GAME_SCREENSHOT_TEMPFILE) -> np.ndarray:
        return self.ss.screenshot(wid=self.wininfo['WindowID'], return_image=return_image, use_play_area=use_play_area, temp_storage=temp_storage)

    def start_game(self):
        command = f'LANG=ja_JP.utf-8 wine {GAME_DIRECTORY}'
        print(command)
        subprocess.Popen(command, shell=True)

    def close(self):
        self.process.close()

    def json(self, path:str):
        json_data = {
            'enemy': self.enemy_list[...],
            'bullet': self.bullet_list[...],
            'laser': self.laser_list[...],
            'stage': self.io_stage.read(),
            'power': self.io_power.read(),
            'score': self.io_score.read(),
            'graze': self.io_graze.read(),
            'life': self.io_life.read(),
            'lifefrag': self.io_lifefrag.read(),
            'bomb': self.io_bomb.read(),
            'bombfrag': self.io_bombfrag.read(),
            'stageframe': self.io_stageframe.read(),
            'key': dataclasses.asdict(self.io_key.read()),
            'pos': self.io_pos.read(),
        }
        if not os.path.exists(path.format(json_data['stageframe'])):
            with open(path.format(json_data['stageframe']), 'w') as f:
                json.dump(json_data, f)
                print('Frame {} dumped'.format(json_data['stageframe']))


if __name__ == '__main__':
    tm = TouhouMaster()
