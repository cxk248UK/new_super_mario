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

save_dir = Path(SAVE_DIR) / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
models = list(save_dir.parent.glob('**/model'))
last_model_path = None

if len(models) > 0:
    models.sort(key=lambda model_path: str(model_path))
    last_model_path = models.pop()

mario = GameAgent(state_dim=(4, 84, 84), action_dim=game_env.action_space.n, save_dir=save_dir,
                  last_model_path=last_model_path)

logger = MetricLogger(save_dir)

episodes = 100
for e in range(episodes):

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

    torch.save(mario.net, save_dir / 'model')

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
