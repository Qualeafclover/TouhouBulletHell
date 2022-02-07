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

# DATASET_PATH = '/home/shin/Desktop/TouhouBulletHell/json_dataset'
DATASET_PATH = 'C:/Users/quale/Desktop/TouhouBulletHell/json_dataset'
DATASET_TTS = 0.1
DATASET_SEED = 104
DATASET_PRELOAD_LEVEL = 0
DATASET_ANGLES = 256
DATASET_DEEP_VISION_DEPTH = 600  # Only used if DATASET_DATA_STYLE is 'deep_vision'
DATASET_TRAIN_BATCH_SIZE = 8
DATASET_TEST_BATCH_SIZE = 8

DATASET_DATA_STYLE = [
    'raw',  # array -> [BATCH, FRAME, IMAGE_HEIGHT, IMAGE_WIDTH]
    'simple_vision',  # stacked lines -> [BATCH, FRAME, ANGLES, [DISTANCE, HIT]]
    'deep_vision',  # TODO stacked full vision lines -> [BATCH, FRAME, VIEW_ANGLES, VIEW_HITS]
][2]

DATASET_STACK_STYLE = [
    'use_all',  # uses all [1, 2, 3, 4] -> [1, 1], [1, 2], [2, 3], [3, 4], [4, 4]
    'skip_dupes',  # skips all duplicating index ref [1, 2, 3, 4] -> [1, 2], [2, 3], [3, 4]
    'skip_all'  # TODO skips all duplicating area of index ref [1, 2, 3, 4] -> [1, 2], [3, 4]
][1]

DATASET_STACKS = 5
DATASET_STACK_FRAME_SKIP = 10
DATASET_SMOOTHEN = 1

#################
# MODEL CONFIGS #
#################

MODEL_SEED = 104
MODEL_VISION = 256
