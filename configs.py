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
DATASET_PRELOAD_LEVEL = 2
DATASET_ANGLES = 256
DATASET_TRAIN_BATCH_SIZE = 8
DATASET_TEST_BATCH_SIZE = 8

DATASET_DATA_STYLE = [
    'raw',  # array TODO
    'simple_vision',  # stacked lines
    'deep_vision',  # stacked full vision lines TODO
][1]

DATASET_STACK_STYLE = [
    'use_all',  # uses all
    'skip_dupes',  # skips all duplicating index ref
    'skip_all'  # skips all duplicating area of index ref TODO
][0]

DATASET_STACKS = 5
DATASET_STACK_FRAME_SKIP = 10
DATASET_SMOOTHEN = 1

#################
# MODEL CONFIGS #
#################

MODEL_SEED = 104
MODEL_VISION = 256
