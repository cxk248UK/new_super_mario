import datetime
from pathlib import Path

import torch

from agent import GameAgent
from environment import init_environment
from learn_log import MetricLogger
from custom_common_dict import SAVE_DIR, FRAME_WIDTH, FRAME_HIGH, FRAME_SKIP, USE_CUDA

print(f"Using CUDA: {USE_CUDA}")
print()

game_env = init_environment()
save_dir = Path(SAVE_DIR) / f'{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}_{game_env.unwrapped.gamename}'
save_dir.mkdir(parents=True)


mario = GameAgent(state_dim=(FRAME_SKIP, FRAME_WIDTH, FRAME_HIGH), action_dim=game_env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 500000


for e in range(0, episodes):

    state = game_env.reset()[0]

    # Play the game!
    while True:
        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        observation, reward, terminated, truncated, info = game_env.step(action)

        game_env.render()

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

    logger.log_episode()

    if (e % 10 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
