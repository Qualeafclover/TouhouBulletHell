import time


class Timer(object):
    def __init__(self, total_steps: int, start=False):
        self.total_steps = total_steps
        self.current_steps = None
        if start:
            self.start()

    def start(self):
        self.start_time = time.perf_counter()
        self.current_steps = 0

    def step(self, steps=1):
        assert self.current_steps is not None, 'Timer has not started.'
        self.current_steps += steps

    def left(self):
        assert self.current_steps is not None, 'Timer has not started.'
        if self.current_steps:
            time_passed = time.perf_counter() - self.start_time
            time_per_step = time_passed / self.current_steps
            remaining_steps = self.total_steps - self.current_steps
            time_left = time_per_step * remaining_steps
            return time_left

        else:
            return float('inf')

    def spent(self):
        assert self.current_steps is not None, 'Timer has not started.'
        if self.current_steps:
            time_passed = time.perf_counter() - self.start_time
            return time_passed

        else:
            return float('inf')
