
import numpy as np

class Cache(object):

    def __init__(self):
        self.cache = {}

    def insert(self, result):
        replay_schedule = result['replay_schedule']
        rs_2d = replay_schedule.transform_to_2d()
        self.cache[rs_2d.tobytes()] = result

    def get(self, replay_schedule):
        rs_2d = replay_schedule.transform_to_2d()
        result = self.cache.get(rs_2d.tobytes())
        if result is not None:
            return result, True
        
        return None, False

    def reset(self):
        self.cache = {}