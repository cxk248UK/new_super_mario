import random

import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

import CNN
from ProjectConf import DefaultProjectConf
from collections import deque

from custom_common_dict import USE_CUDA, EXPERT_DATA_MEMORY


def chose_action_from_network_output_with_softmax(network_output):
    softmax = nn.Softmax(dim=1)
    network_output = softmax(network_output)
    network_output = torch.multinomial(network_output, 1)
    return network_output


class GameAgent:
    def __init__(self, state_dim, action_dim, save_dir,
                 conf=DefaultProjectConf()):
        # check device
        self.device = "cuda" if USE_CUDA else "cpu"
        #   game agent setting
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        net_class = getattr(CNN, conf.net_name)

        self.net = net_class(input_dim=self.state_dim, output_dim=self.action_dim)

        if conf.imitation:
            self.exploration_rate = conf.imitation_exploration_rate
            self.exploration_rate_decay = conf.imitation_exploration_rate_decay
        else:
            self.exploration_rate = conf.exploration_rate
            self.exploration_rate_decay = conf.exploration_rate_decay
        self.exploration_rate_min = conf.exploration_rate_min
        self.curr_step = 0

        self.save_every = conf.save_every

        #     cache and recall setting
        self.memory = deque(maxlen=5000)
        self.batch_size = conf.batch_size

        #     learn rate for Q_learning
        self.gamma = conf.gamma

        #     CNN setting
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        #       learning setting
        self.min_experience_num = conf.min_experience_num
        self.learn_every = conf.learn_every
        self.sync_every = conf.sync_every

        self.imitation_flag = conf.imitation

        # load model if possible
        if conf.checkpoint:
            self.load(conf.checkpoint)

        self.net = self.net.to(device=self.device)

        self.conf = conf

    def act(self, state, play=False):
        if play:
            # 根据模型预测探索
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_value = self.net(state, model='online')
            action_idx = chose_action_from_network_output_with_softmax(action_value).item()
            return action_idx

        if np.random.rand() < self.exploration_rate:
            # 随机探索
            action_idx = np.random.randint(self.action_dim)
        else:
            # 根据模型预测探索
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_value = self.net(state, model='online')
            action_idx = chose_action_from_network_output_with_softmax(action_value).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        if self.imitation_flag:
            reward = torch.tensor([0.0])
        else:
            reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        # soft q imitation learning recall
        if self.imitation_flag:
            exploration_batch = random.sample(self.memory, int(self.batch_size / 2))
            expert_batch = random.sample(EXPERT_DATA_MEMORY, self.batch_size - int(self.batch_size / 2))
            batch = exploration_batch + expert_batch
        # classical soft q learning recall
        else:
            batch = random.sample(self.memory, self.batch_size)

        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        state = state.to(device=self.device)
        next_state = next_state.to(device=self.device)
        action = action.to(device=self.device)
        reward = reward.to(device=self.device)
        done = done.to(device=self.device)

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = chose_action_from_network_output_with_softmax(next_state_Q)
        best_action = torch.squeeze(best_action)
        next_Q = self.net(next_state, model='target')[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.min_experience_num:
            return None, None

        if len(self.memory) < self.batch_size:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss

    def save(self):
        save_path = (
                self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")
        return save_path

    def load(self, load_path):
        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate

    def switch_imitation(self):
        self.imitation_flag = False
        self.exploration_rate_decay = self.conf.exploration_rate_decay
        self.exploration_rate = max(self.conf.exploration_rate ** self.curr_step, self.exploration_rate_min)
        self.memory.clear()
