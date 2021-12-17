####################
# TRAINING CONFIGS #
####################

TRAIN_EPOCHS = 5
# TRAIN_DATA_PATH = '/home/shin/Desktop/TouhouBulletHell/json_dataset'
TRAIN_DATA_PATH = 'C:/Users/quale/Desktop/TouhouBulletHell/json_dataset'

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
DATASET_TTS = 0.2
DATASET_SEED = 104
DATASET_PRELOAD_LEVEL = 0
DATASET_ANGLES = 256
DATASET_BATCH_SIZE = 8

# Unimplemented
DATASET_DATA_STYLE = [
    'raw',
    'simple vision',
    'deep vision',
][1]

#################
# MODEL CONFIGS #
#################

MODEL_SEED = 104
MODEL_VISION = 256
