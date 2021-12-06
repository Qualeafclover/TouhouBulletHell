import tensorflow as tf
import os

from .checkpoint_configs import *


class Checkpoint(object):
    def __init__(self, log_dir=CHECKPOINT_LOGS, save_best=CHECKPOINT_SAVE_BEST):
        self.log_dir = log_dir
        self.save_best = save_best
        self.best_loss = float('inf')

    def save(self, model: tf.keras.Model, epoch_num: int, loss: float) -> str:
        self.best_loss = min(self.best_loss, loss)
        if (self.save_best and self.best_loss == loss) or (not self.save_best):
            save_dir = os.path.join(self.log_dir, f'checkpoint-epoch{epoch_num:03d}-loss{loss:.03f}')
            model.save_weights(save_dir)
            return save_dir
