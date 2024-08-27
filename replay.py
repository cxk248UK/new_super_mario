import json
import random

import torch
from tensordict import TensorDict
from torch import nn

from CNN import MiniCnnModel
from custom_common_dict import USE_CUDA
from environment import init_environment


def first_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x


def train():
    print(f"Using CUDA: {USE_CUDA}")
    print()

    game_env = init_environment(render=False)
    net = MiniCnnModel((4, 84, 84), 3)
    ckp = torch.load('default_cnn_144:/mario_net_20.chkpt', map_location='cpu')
    state_dict = ckp.get('model')
    net.load_state_dict(state_dict)

    max_reward = 0
    max_reward_actions = []

    for i in range(1000):

        episodes_record = []

        total_reward = 0

        state = game_env.reset()[0]

        last_time = 0
        last_time_count = 0

        # Play the game!
        while True:
            # Run agent on the state
            state = torch.tensor(state.__array__())
            state = state.unsqueeze(0)
            if random.random() < 0.1:
                action = game_env.action_space.sample()
            else:
                action = net(state)
                softmax = nn.Softmax(dim=1)
                action = softmax(action)
                action = torch.multinomial(action, 1)
                action = action.item()

            episodes_record.append(int(action))
            # Agent performs action
            observation, reward, terminated, truncated, info = game_env.step(action)

            lives = info.get('lives')
            time = info.get('time')
            if time != last_time:
                last_time = time
                last_time_count = 0
            else:
                last_time_count += 1

            done = terminated or truncated or (lives < 2) or (last_time_count > 10)

            state = observation

            total_reward += reward

            # Check if end of game
            if done:
                break

        if total_reward > max_reward:
            max_reward = total_reward
            max_reward_actions = episodes_record
            print(f'episodes :{i} -- reward: {max_reward}')

    json.dump(max_reward_actions, open('max_reward_actions.json', 'w'))


if __name__ == '__main__':
    train()
