####################
# TRAINING CONFIGS #
####################

TRAIN_EPOCHS = 5
TRAIN_DATA_PATH = '/home/shin/Desktop/TouhouBulletHell/json_dataset'
# TRAIN_DATA_PATH = 'C:/Users/quale/Desktop/TouhouBulletHell/json_dataset'

#######################
# TENSORBOARD CONFIGS #
#######################

TENSORBOARD_LOG = 'tensorboard_log'
TENSORBOARD_RESET = True
TENSORBOARD_LAUNCH = True

##################
# LOGGER CONFIGS #
##################

LOGGER_DIRECTORY = 'logs'
LOGGER_VERBOSE = True

######################
# CHECKPOINT CONFIGS #
######################

CHECKPOINT_LOGS = 'checkpoint'
CHECKPOINT_SAVE_BEST = False

###################
# DATASET CONFIGS #
###################

DATASET_PATH = '/home/shin/Desktop/TouhouBulletHell/json_dataset'
# DATASET_PATH = 'C:/Users/quale/Desktop/TouhouBulletHell/json_dataset'
DATASET_TTS = 0.1
DATASET_SEED = 104
DATASET_PRELOAD_LEVEL = 0  # level 3 and above does not support timed preloading
DATASET_ANGLES = 256
DATASET_DEEP_VISION_DEPTH = 256  # Only used if DATASET_DATA_STYLE is 'deep_vision'
DATASET_TRAIN_BATCH_SIZE = 1
DATASET_TEST_BATCH_SIZE = 1

DATASET_DATA_STYLE = [
    'raw',  # array -> [BATCH, IMAGE_HEIGHT, IMAGE_WIDTH, FRAME]
    'simple_vision',  # stacked lines -> [BATCH, ANGLES, [DISTANCE, HIT, FRAME]]
    'deep_vision',  # stacked full vision lines -> [BATCH, VIEW_ANGLES, VIEW_HITS, FRAME]
][2]

DATASET_STACK_STYLE = [
    'use_all',  # uses all [1, 2, 3, 4] -> [1, 1], [1, 2], [2, 3], [3, 4], [4, 4]
    'skip_dupes',  # skips all duplicating index ref [1, 2, 3, 4] -> [1, 2], [2, 3], [3, 4]
    'skip_all'  # TODO skips all duplicating area of index ref [1, 2, 3, 4] -> [1, 2], [3, 4]
][1]

DATASET_STACKS = 3
DATASET_STACK_FRAME_SKIP = 4
DATASET_SMOOTHEN = 1  # TODO smoothen key outputs based on time. Might not be necessary, more experiments needed

#################
# MODEL CONFIGS #
#################

MODEL_SEED = 104
MODEL_VISION = 256
