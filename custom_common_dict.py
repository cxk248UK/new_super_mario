import os.path
from collections import deque

import torch
import json
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, LazyTensorStorage
import shutil

USE_CUDA = torch.cuda.is_available()

USE_DEVICE = 'cpu'
if USE_CUDA:
    USE_DEVICE = 'cuda'

# 允许的动作组合,单按键或两个按键的组合
# 按键列表['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
OPERATE_BUTTONS = ['B', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
MENU_BUTTONS = ['SELECT', 'START']
GAME_NAME = 'SuperMarioBros-Nes'
TRAIN_RENDER = 'None'
PLAY_RENDER = 'human'
TRAINED_MODEL_PATH = 'trained_mario.chkpt'

ALLOW_ACTION_COMBOS = [['RIGHT'], ['RIGHT', 'A'], ['A']]
# for button in OPERATE_BUTTONS:
#     ALLOW_ACTION_COMBOS.append([button])
#
# for first_button_index in range(len(OPERATE_BUTTONS)):
#     for second_button_index in range(first_button_index + 1, len(OPERATE_BUTTONS)):2
#         ALLOW_ACTION_COMBOS.append([OPERATE_BUTTONS[first_button_index], OPERATE_BUTTONS[second_button_index]])
#
# for menu in MENU_BUTTONS:
#     ALLOW_ACTION_COMBOS.append([menu])

FRAME_WIDTH = 84
FRAME_HIGH = 84
FRAME_SKIP = 4
NET_INPUT_SHAPE = (FRAME_SKIP, FRAME_WIDTH, FRAME_HIGH)
NET_OUTPUT_DIM = len(ALLOW_ACTION_COMBOS)

# 降采样后观察空间形状
OBSERVATION_SHAP = (FRAME_WIDTH, FRAME_HIGH)

# 保存文件路径（开发和测试环境不同）
# COLAB 保存地址
# SAVE_DIR = '../drive/MyDrive/Colab Notebooks/checkpoints'
# 本地保存地址
SAVE_DIR = 'checkpoints'

#
# EXPERT_DATA_FILE = open('expert_data','r')
# EXPERT_DATA = EXPERT_DATA_FILE.readlines()
# EXPERT_DATA_FILE.close()
# EXPERT_DATA = json.load(EXPERT_DATA)

MAX_EPISODES = 50000
MAX_IMITATION_EPISODES = 10000
MAX_EXPLORATION_EPISODES = 40000

EXPLORATION_RATE_DECAY = 0.99999975
MIN_EXPLORATION_RATE = 0.1

# EXPERT_DATA = []
#
# GENERATE_EXPERT_DATA = []

EXPERT_DATA_MEMORY = TensorDictReplayBuffer(storage=LazyTensorStorage(2600, device=torch.device("cpu")))

expert_data_prefix = ''

if os.path.exists('../drive/MyDrive/new_super_mario/'):
    expert_data_prefix = '../drive/MyDrive/new_super_mario/'

for i in range(1, 6):
    EXPERT_DATA = torch.load(f'{expert_data_prefix}expert_data_{i}')
    for expert_data in EXPERT_DATA:
        EXPERT_DATA_MEMORY.add(expert_data)
#
# for i in range(1, 11):
#     EXPERT_DATA = torch.load(f'generate_expert_data_{i}')
#     for expert_data in EXPERT_DATA:
#         EXPERT_DATA_MEMORY.add(expert_data)
