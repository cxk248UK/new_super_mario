import inspect
import sys

from torch import nn
import torch
from custom_common_dict import EXPERT_DATA_MEMORY
import CNN
from Agent import chose_action_from_network_output_with_softmax
import numpy as np
import json


def recall():
    """
    Retrieve a batch of experiences from memory
    """
    batch = EXPERT_DATA_MEMORY.sample(32)
    state, next_state, action, reward, done = (batch.get(key) for key in
                                               ("state", "next_state", "action", "reward", "done"))
    return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


def learn(model_name='MiniCnnModel'):
    model = getattr(CNN, model_name)
    test_model = model((4, 84, 84), 3)

    optimizer = torch.optim.Adam(test_model.parameters(), lr=0.00025)
    loss_fn = torch.nn.SmoothL1Loss()

    loss_array = []
    mean_loss = []

    for e in range(20000):
        state, next_state, action, reward, done = recall()
        td_est = test_model(state, model='online')[
            np.arange(0, 32), action
        ]

        next_state_Q = test_model(next_state, model='online')
        best_action = chose_action_from_network_output_with_softmax(next_state_Q)

        best_action = torch.squeeze(best_action)
        next_Q = test_model(next_state, model='online')[
            np.arange(0, 32), best_action
        ]
        target = (reward + (1 - done.float()) * 0.9 * next_Q).float()

        loss = loss_fn(td_est, target)
        loss_array.append(loss.item())

        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 100 == 0:
            mean_loss.append(np.round(np.mean(loss_array[-100:]), 3))
            print(mean_loss[-1])

    json_file = open(f'./BC_result/{model_name}.json', 'w')

    json.dump(mean_loss, json_file)


if __name__ == '__main__':
    clsmembers = inspect.getmembers(sys.modules['CNN'], inspect.isclass)
    for cls in clsmembers:
        if cls[0].find('Mini') != -1:
            learn(cls[0])
