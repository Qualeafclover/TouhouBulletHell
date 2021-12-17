from configs import *

import os
import datetime


class Logger(object):
    def __init__(self, logdir=LOGGER_DIRECTORY, verbose=LOGGER_VERBOSE):
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        now = datetime.datetime.now()
        log_filename = now.strftime('-%Y_%m_%d-%H_%M_%S-.txt')
        self.verbose = verbose
        self.log_path = os.path.join(logdir, log_filename)
        self.log_start_time()

    def time_log(self, log_content: str, logtype='I'):
        now = datetime.datetime.now()
        now = now.strftime('%Y-%b-%d %H:%M:%S')
        with open(self.log_path, 'a') as f:
            f.write("{} ({}) {}\n".format(now, logtype, log_content))
        print("{} ({}) {}".format(now, logtype, log_content))
        return now

    def log(self, *args, logtype='I', **kwargs):
        args_kwargs = ' | '.join(args + tuple(f'{key}: {kwargs[key]}' for key in kwargs))
        self.time_log(args_kwargs, logtype=logtype)

    def log_start_time(self):
        self.time_log('Started program instance.')

