import os
import json
import numpy as np
import time
import sys
from environment import Env, DISCRETE_CONTROL
from agents import DQN

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

ENV_PATH = 'maps/environments/env.png'

EPISODES = 5000
MEMORY_CAPACITY = 2048

# python -m visdom.server
from visdom import Visdom
viz = Visdom(env='DQN Train on Env')
reward_opt = {  # options for visualization of reward
    'title' :'Reward',
    "xlabel":'Episode',
    "ylabel":'Reward',
}

if __name__ == '__main__':

    state_size = 40
    action_size = 5

    env = Env(action_type=DISCRETE_CONTROL, action_size=action_size, env_path=ENV_PATH)

    agent = DQN(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    memory_index = 0
    memory_length = 0

    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
        score = 0
        for t in range(agent.episode_step):
            action = agent.getAction(state)

            next_state, reward, done = env.step(action)
            memory_length += 1

            if memory_length <= MEMORY_CAPACITY:
                agent.appendMemory(state, action, reward, next_state, done)
            else:
                memory_index = memory_index % MEMORY_CAPACITY
                agent.memory[memory_index] = (state, action, reward, next_state, done)

            memory_index += 1
            # print('append memory: ', state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)

            score += reward
            state = next_state
            get_action_data = [action, score, reward]

            if e % 50 == 0:
                agent.model.save(agent.dirPath + str(e) + '.h5')
                with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)

            if t >= 500:
                print("Time out!!")
                done = True

            if done:
                print [score, np.max(agent.q_value)]
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                print('Ep: ', e, 'Score: ', score, 'memory: ', len(agent.memory), 'epsilon: ', agent.epsilon)

                # print('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                #               e, score, len(agent.memory), agent.epsilon, h, m, s)

                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                print("UPDATE TARGET NETWORK")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        viz.line(X=[e], Y=[score], win='Training', opts=reward_opt,
                 update=None if e > MEMORY_CAPACITY else 'append')
