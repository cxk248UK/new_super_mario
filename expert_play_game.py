from tensordict import TensorDict

from environment import init_environment
from custom_common_dict import TRAINED_MODEL_PATH, USE_CUDA, FRAME_SKIP, FRAME_WIDTH, FRAME_HIGH
from Agent import GameAgent
import torch

game_env = init_environment(render=True)

state = game_env.reset()[0]
total_reward = 0
action_record = []
expert_data = []

last_time = 0
last_time_count = 0

while True:
    # Run agent on the state
    game_env.render()
    action = input("Enter a number:")
    if action.isnumeric():
        action = int(action)
        if action > 2:
            continue
    else:
        continue

    # Agent performs action
    next_state, reward, terminated, truncated, info = game_env.step(action)

    lives = info.get('lives')
    time = info.get('time')

    if time != last_time:
        last_time = time
        last_time_count = 0
    else:
        last_time_count += 1

    done = terminated or truncated or (lives < 2) or (last_time_count > 10)

    state = state.__array__()
    next_state = next_state.__array__()

    state = torch.tensor(state)
    next_state = torch.tensor(next_state)
    action = torch.tensor([action])
    reward = torch.tensor([reward])
    done = torch.tensor([done])
    expert_data.append(
        TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}))

    # Update state
    state = next_state

    # Check if end of game
    if done:
        break

expert_data = []

for i in range(1, 6):
    expert_data.extend(torch.load(f'expert_data_{i}'))

expert_data[100].get('reward')
