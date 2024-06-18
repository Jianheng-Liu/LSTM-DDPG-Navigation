import os
import json
import numpy as np
import time
import sys
from environment import Env
from agents import DQN

import random

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

ENV_PATH = 'maps/environments/env3.png'

EPISODES = 200
MEMORY_CAPACITY = 1024

if __name__ == '__main__':

    state_size = 40
    action_size = 5

    env = Env(action_size, ENV_PATH)

    agent = DQN(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    env.reset()

    for i in range(100):
        env.get_state()
        action = random.randrange(action_size)
        # print('action: ', action)
        env.step(action=action)
