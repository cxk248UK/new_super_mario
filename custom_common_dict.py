import torch
import json

USE_CUDA = torch.cuda.is_available()

# 允许的动作组合,单按键或两个按键的组合
# 按键列表['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
OPERATE_BUTTONS = ['B', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
MENU_BUTTONS = ['SELECT', 'START']
GAME_NAME = 'SuperMarioBros-Nes'
TRAIN_RENDER = 'None'
PLAY_RENDER = 'human'
TRAINED_MODEL_PATH = 'mario_net_31.chkpt'

ALLOW_ACTION_COMBOS = [['RIGHT'], ['RIGHT', 'A']]
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

