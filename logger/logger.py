from configs import *

import os
import datetime
import logging


# Please forgive me for stacking logger classes
class Logger(object):
    def __init__(self, logdir=LOGGER_DIRECTORY, verbose=LOGGER_VERBOSE):
        if logdir is None:
            self.no_log = True
        else:
            self.no_log = False
            if not os.path.isdir(logdir):
                os.mkdir(logdir)
        now = datetime.datetime.now()
        log_filename = now.strftime('-%Y_%m_%d-%H_%M_%S-.txt')
        self._logger = logging.Logger('Logger')
        self.verbose = verbose
        self.log_path = os.path.join(logdir, log_filename)
        self.log_start_time()

    def _time_log(self, log_content: str, logtype='I'):
        log_level = {'N': 0, 'D': 10, 'I': 20, 'W': 30, 'E': 40, 'C': 50}[logtype]
        now = datetime.datetime.now()
        now = now.strftime('%Y-%b-%d %H:%M:%S')
        log_text = f"{now} ({logtype}) {log_content}"
        if not self.no_log:
            with open(self.log_path, 'a') as f:
                f.write(f'{log_text}\n')
        self._logger.log(log_level, log_text)
        if log_level < 30 and self.verbose:
            print(log_text)
        return now

    def log(self, *args, logtype='I', **kwargs):
        args_kwargs = ' | '.join(args + tuple(f'{key}: {kwargs[key]}' for key in kwargs))
        self._time_log(args_kwargs, logtype=logtype)

    def log_start_time(self):
        self._time_log('Started logger instance.')


if __name__ == '__main__':
    logger = Logger()
    logger.log('Hello!', logtype='W')
