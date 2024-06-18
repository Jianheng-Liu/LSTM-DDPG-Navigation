import os
import torch
import numpy as np
import random
import time
import sys
from environment import Env, CONTINUOUS_CONTROL
from agents import DDPG
from math import pi

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

ENV_PATH = 'maps/environments/env3.png'

LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement

MEMORY_CAPACITY = 10000
BATCH_SIZE = 256
epochs = 2048
steps = 250
learnStart = 8
update_freq = 64

EXPLORE = 512  # frames over which to anneal epsilon
INITIAL_EPSILON = 1  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
explorationRate = INITIAL_EPSILON
current_epoch = 0
loadsim_seconds = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# python -m visdom.server
from visdom import Visdom

visualization = True
if visualization:
    viz = Visdom(env='DDPG Train')
reward_opt = {  # options for visualization of reward
    'title': 'Reward',
    "xlabel": 'Episode',
    "ylabel": 'Reward',
}
train_opt = {  # options for visualization of training reward
    'title': 'Training',
    "xlabel": 'Episode',
    "ylabel": 'Reward',
}

epsilon_opt = {  # options for visualization of goal number
    'title' :'Epsilon',
    "xlabel":'Episode',
    "ylabel":'Num',
}

if __name__ == '__main__':

    state_size = 40
    scan_size = 36
    pos_size = 4
    action_size = 1
    action_bound = 1
    env = Env(action_type=CONTINUOUS_CONTROL, action_size=action_size, env_path=ENV_PATH)

    # agent = ReinforceAgent(state_size, action_size)
    agent = DDPG(state_size, scan_size, pos_size, action_size, action_bound)
    load_network = False
    load_path = '/home/casia/navigation/ddpg_models/Episode700-CReward-595.42-totalstep134578-explore0.009011.pth'
    if load_network:
        print "load model"
        agent.load_state_dict(torch.load(load_path))

    agent = agent.to(device)  # transfer mobel from CPU  to the GPU
    scores, episodes = [], []
    start_time = time.time()
    stepCounter = 0
    for episode in xrange(current_epoch + 1, epochs + 1, 1):
        state = env.reset()
        scores, episodes = [], []
        reward_sum = 0
        done = False
        collision_num = 0
        goal_num = 0
        # print("epoch+1!\n")
        if episode == learnStart:
            print("Starting learning")
        # number of timesteps
        for t in xrange(steps):
            # use GPU to  train the data
            state_scan = state[:scan_size]
            state_pos = state[-pos_size:]

            action = agent.choose_action(state_scan, state_pos)
            # print("action: ", action)
            # convert action from (type = gpu) to (type = cpu)
            action = action.data.cpu().numpy()
            # add exploration noise
            # action = np.clip(np.random.normal(action, explorationRate), 0, 1)
            i = random.random()
            if i < explorationRate:
                action[0] = random.random()
            # remain 2 decimal places
            # action[0] = round(action[0] * 0.26, 2)  #linear: 0~0.26
            action[0] = round(((action[0] * 2 - 1) * pi / 4), 2)  # angular:-pi/4, pi/4

            # agents interact with environment
            next_state, reward, done = env.step(action[0])

            # store the transitions
            agent.store_transition(state, action, reward, next_state)

            # update the state
            # env._flush(force=True)
            state = next_state
            reward_sum += reward

            # We reduced the epsilon gradually
            if explorationRate > FINAL_EPSILON and episode > learnStart and stepCounter % 20 == 0:
                explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            if episode >= learnStart and stepCounter % update_freq == 0:
                agent.learn()
                print("Update")

            if done:
                # agent.updateTargetModel()
                episodes.append(t)
                print('Ep: ', episode, 'Score: ', reward_sum, 'memory: ', agent.pointer, 'epsilon: ', explorationRate)
                break

            stepCounter += 1

        if visualization:
            viz.line(X=[episode], Y=[reward_sum], win='Training', opts=reward_opt,
                     update=None if episode > MEMORY_CAPACITY else 'append')
            viz.line(X=[episode], Y=[explorationRate], win='Epsilon', opts=epsilon_opt,
                     update=None if episode > MEMORY_CAPACITY else 'append')
        if visualization and episode >= learnStart:
            viz.line(X=[episode], Y=[reward_sum], win='Training', opts=train_opt,
                     update=None if episode > MEMORY_CAPACITY else 'append')
        if episode % 100 == 0:
            # save model weights and monitoring data every 100 episodes.
            torch.save(agent.state_dict(),
                       '/home/casia/navigation/ddpg_models/Episode%d-CReward%.2f-totalstep%d-explore%f.pth' % (
                       episode, reward_sum, stepCounter, explorationRate))

