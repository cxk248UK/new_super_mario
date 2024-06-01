import datetime
from pathlib import Path

import torch

from agent import GameAgent
from environment import game_env
from learn_log import MetricLogger
from custom_common_dict import SAVE_DIR

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path(SAVE_DIR) / f'{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}_{game_env.gamename}'
save_dir.mkdir(parents=True)
models = list(save_dir.parent.glob('**/model'))
last_log = list(save_dir.parent.glob('*SuperMarioBros3-Nes/log'))
last_model_path = None
last_e = None

if len(models) > 0:
    models.sort(key=lambda model_path: str(model_path))
    last_model_path = models.pop()

if len(last_log) > 0:
    last_log.sort(key=lambda model_path: str(model_path))
    try:
        last_e = int(open(last_log.pop()).readlines().pop().split()[0])
    except IOError or TypeError:
        print('try to get last episodes but fail')

mario = GameAgent(state_dim=(4, 84, 84), action_dim=game_env.action_space.n, save_dir=save_dir,
                  last_model_path=last_model_path)

logger = MetricLogger(save_dir)

episodes = 500000

if last_e and last_model_path:
    start_e = last_e
    print(f'Start from last episodes: {start_e}')
else:
    start_e = 0
    print('can not find last episodes. start from 0')
for e in range(start_e, episodes):

    state = game_env.reset()[0]

    # Play the game!
    step = 0
    while step <= 9000:
        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        observation, reward, terminated, truncated, info = game_env.step(action)

        # game_env.render()

        # Remember
        mario.cache(state, observation, action, reward, int(terminated or truncated))

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = observation

        # Check if end of game
        if terminated or truncated:
            break
        step += 1

    logger.log_episode()

    if (e % 10 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
        torch.save(mario.net, save_dir / 'model')
