# There is another confusingly similar ～ character, be careful.
# Some configurations are different on the 1.00b game version.
GAME_WINDOW_NAME = "東方輝針城　～ Double Dealing Character. ver 1.00a"
GAME_PROCESS_NAME = 'th14.exe'
GAME_DIRECTORY = '"/home/shin/.wine/drive_c/Program Files (x86)/上海アリス幻樂団/東方輝針城/th14.exe"'

GAME_WORKING_ABS_W = (lambda w: int(round(w/20*1)))
GAME_WORKING_ABS_H = (lambda h: int(round(h/30*1)))
GAME_WORKING_W = (lambda w: int(round(w/20*12)))
GAME_WORKING_H = (lambda h: int(round(h/30*28)))

GAME_SCREENSHOT_TEMPFILE = '/home/shin/Desktop/TouhouBulletHell/game_master/temp.bmp'

import ctypes

GAME_ENEMY_LIST_POINTER = 0x4DD524
GAME_ENEMY_LIST_CTYPE = ctypes.c_uint32
GAME_ENEMY_LIST_POINTER_OFFSET = 0xD4
GAME_ENEMY_LIST_NEXT_ITEM_OFFSET = 0x8
GAME_ENEMY_LIST_BREAKPOINT = 0x0
GAME_ENEMY_LIST_LOOKUP = {
    # Stage 4 yinyang
    # 'x': (-0x21C, ctypes.c_float), 'y': (-0x218, ctypes.c_float),
    # 'speed': (-0x1E8, ctypes.c_float), 'direction': (-0x1E4, ctypes.c_float),

    'x': (-0x260, ctypes.c_float), 'y': (-0x25C, ctypes.c_float),
    'speed': (-0x248, ctypes.c_float), 'direction': (-0x244, ctypes.c_float),

    'diameter': (-0x1D0, ctypes.c_float),
    'type': (-0x80, ctypes.c_uint32),
}

GAME_BULLET_LIST_POINTER = 0x4DD510
GAME_BULLET_LIST_CTYPE = ctypes.c_uint32
GAME_BULLET_LIST_POINTER_OFFSET = 0x80
GAME_BULLET_LIST_NEXT_ITEM_OFFSET = 0x4
GAME_BULLET_LIST_BREAKPOINT = 0x0
GAME_BULLET_LIST_LOOKUP = {
    'x': (+0xBB0, ctypes.c_float), 'y': (+0xBB4, ctypes.c_float),
    'x_speed': (+0xBBC, ctypes.c_float), 'y_speed': (+0xBC0, ctypes.c_float),
    'radius': (+0xBD0, ctypes.c_float), 'radius_mult': (+0x13AC, ctypes.c_float)
}

GAME_LASER_LIST_POINTER = 0x4DD644
GAME_LASER_LIST_CTYPE = ctypes.c_uint32
GAME_LASER_LIST_POINTER_OFFSET = 0x5D0
GAME_LASER_LIST_NEXT_ITEM_OFFSET = 0x4
GAME_LASER_LIST_BREAKPOINT = 0x0
GAME_LASER_LIST_LOOKUP = {
    'type': (+0x0, ctypes.c_uint32),
    'counter_1': (+0x20, ctypes.c_float), 'counter_2': (+0x34, ctypes.c_float),
    'counter_3': (+0x48, ctypes.c_float), 'counter_4': (+0x7C, ctypes.c_float),
    'x': (+0x54, ctypes.c_float), 'y': (+0x58, ctypes.c_float),
    'x_speed': (+0x60, ctypes.c_float), 'y_speed': (+0x64, ctypes.c_float),
    'direction': (+0x6c, ctypes.c_float), 'length': (+0x70, ctypes.c_float),
    'thickness': (+0x74, ctypes.c_float), 'growth': (+0x78, ctypes.c_float),
}

GAME_INFLIFE = 0x44F617
GAME_INFLIFE_CTYPE = ctypes.c_char
GAME_INFLIFE_ENABLE = b'\x90'
GAME_INFLIFE_DISABLE = b'\x48'
GAME_INFLIFE_DICT = {
    b'\x90': True,
    b'\x48': False,
}

GAME_INFBOMB = 0x412173
GAME_INFBOMB_CTYPE = ctypes.c_char
GAME_INFBOMB_ENABLE = b'\x90'
GAME_INFBOMB_DISABLE = b'\x48'
GAME_INFBOMB_DICT = {
    b'\x90': True,
    b'\x48': False,
}

GAME_AUTOBOMB = 0x44DEC4
GAME_AUTOBOMB_CTYPE = ctypes.c_char
GAME_AUTOBOMB_ENABLE = b'\xc6'
GAME_AUTOBOMB_DISABLE = b'\xf6'
GAME_AUTOBOMB_DICT = {
    b'\xc6': True,
    b'\xf6': False,
}

GAME_INVULN = 0x44F877
GAME_INVLUN_CTYPE = ctypes.c_char
GAME_INVLUN_ENABLE = b'\x01'
GAME_INVLUN_DISABLE = b'\x04'
GAME_INVULN_DICT = {
    b'\x01': True,
    b'\x04': False,
}

GAME_GAMESTATE = 0x4F9AA8
GAME_GAMESTATE_CTYPE = ctypes.c_char
GAME_GAMESTATE_DICT = {
    b'\x00': 'pause',
    b'\x01': 'menu',
    b'\x02': 'game',
}

GAME_STAGE = 0x4F7884
GAME_STAGE_CTYPE = ctypes.c_char
GAME_STAGE_DICT = {
    b'\x00': 0,
    b'\x01': 1,
    b'\x02': 2,
    b'\x03': 3,
    b'\x04': 4,
    b'\x05': 5,
    b'\x06': 6,
    b'\x07': 7,
}

GAME_POWER = 0x4F7838
GAME_POWER_CTYPE = ctypes.c_uint32

GAME_SCORE = 0x4F7810
GAME_SCORE_CTYPE = ctypes.c_uint32

GAME_GRAZE = 0x4F7820
GAME_GRAZE_CTYPE = ctypes.c_uint32

GAME_LIFE = 0x4F7844
GAME_LIFE_CTYPE = ctypes.c_uint32

GAME_LIFEFRAG = 0x4F7848
GAME_LIFEFRAG_CTYPE = ctypes.c_uint32

GAME_BOMB = 0x4F7850
GAME_BOMB_CTYPE = ctypes.c_uint32

GAME_BOMBFRAG = 0x4F7854
GAME_BOMBFRAG_CTYPE = ctypes.c_uint32

GAME_STAGEFRAME = 0x4F7890
GAME_STAGEFRAME_CTYPE = ctypes.c_uint32

GAME_EDI_FIND = 0x4DD65C
GAME_EDI_CTYPE = ctypes.c_uint32
GAME_EDI_X_OFFSET = 0x05E0
GAME_EDI_Y_OFFSET = 0x05E4
GAME_EDI_R_OFFSET = 0x18308
GAME_R_MULT = 0x4C2EC8
GAME_EDI_XYR_CTYPE = ctypes.c_float

GAME_KEYS_ADDR = 0x4D8A90
GAME_KEYS_CTYPE = ctypes.c_uint32
GAME_KEYS_ENCODER = '01345679'

GAME_AIMOVE = 0x455257
GAME_AIMOVE_CTYPE = ctypes.c_uint64
GAME_AIMOVE_ENABLE = 0x94afe89090909090
GAME_AIMOVE_DISABLE = 0x94afe8004d8a90a3
GAME_AIMOVE_DICT = {
    int(0x94afe89090909090): True,
    int(0x94afe8004d8a90a3): False,
}
