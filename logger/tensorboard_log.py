import tensorflow as tf
from configs import *

from tensorboard import program
from datetime import datetime
import shutil
import os


class TensorboardLogger(object):
    def __init__(self, writer_names=('train', 'test')):
        if TENSORBOARD_RESET:
            try:
                shutil.rmtree(TENSORBOARD_LOG)
            except FileNotFoundError:
                pass
        if TENSORBOARD_LAUNCH:
            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, '--logdir', TENSORBOARD_LOG])
            self.url = self.tb.launch()
        else:
            self.tb, self.url = None, None

        self.summary_writers = {}
        for writer_name in writer_names:
            log_dir = os.path.join(
                TENSORBOARD_LOG,
                str(datetime.now().time().replace(microsecond=0)).replace(':', '-') + '_' + writer_name)
            self.summary_writers[writer_name] = tf.summary.create_file_writer(logdir=log_dir)

    def write_summary(self, writer: str, step: int, **kwargs):
        with self.summary_writers[writer].as_default(step=step):
            for key in kwargs:
                tf.summary.scalar(key, kwargs[key])
