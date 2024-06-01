# 允许的动作组合,单按键或两个按键的组合
# 按键列表['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
OPERATE_BUTTONS = ['B', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
MENU_BUTTONS = ['SELECT', 'START']

ALLOW_ACTION_COMBOS = []
for button in OPERATE_BUTTONS:
    ALLOW_ACTION_COMBOS.append([button])

for first_button_index in range(len(OPERATE_BUTTONS)):
    for second_button_index in range(first_button_index + 1, len(OPERATE_BUTTONS)):
        ALLOW_ACTION_COMBOS.append([OPERATE_BUTTONS[first_button_index], OPERATE_BUTTONS[second_button_index]])

for menu in MENU_BUTTONS:
    ALLOW_ACTION_COMBOS.append([menu])

# 降采样后观察空间形状
OBSERVATION_SHAP = (84, 84)

# 保存文件路径（开发和测试环境不同）
# COLAB 保存地址
SAVE_DIR = '../drive/MyDrive/Colab Notebooks/checkpoints'
# 本地保存地址
# SAVE_DIR = 'checkpoints'
