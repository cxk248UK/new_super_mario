import datetime
from pathlib import Path

from agent import GameAgent
from environment import init_environment
from learn_log import MetricLogger
from custom_common_dict import SAVE_DIR, FRAME_WIDTH, FRAME_HIGH, FRAME_SKIP, USE_CUDA
import multiprocessing
import sys


def train(flag=0):
    print(f"Using CUDA: {USE_CUDA}")
    print(f'flag -- {flag}')

    game_env = init_environment()
    save_dir = Path(SAVE_DIR) / f'{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}_{game_env.unwrapped.gamename}'
    save_dir.mkdir(parents=True)

    mario = GameAgent(state_dim=(FRAME_SKIP, FRAME_WIDTH, FRAME_HIGH), action_dim=game_env.action_space.n,
                      save_dir=save_dir, flag=flag)

    logger = MetricLogger(save_dir)

    episodes = 40000

    for e in range(0, episodes):

        state = game_env.reset()[0]

        last_time = 0
        last_time_count = 0

        # Play the game!
        while True:
            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            observation, reward, terminated, truncated, info = game_env.step(action)

            if flag == 1 and e >= 2000 and mario.flag == 1:
                mario.flag = 0
                mario.memory.empty()
            lives = info.get('lives')
            time = info.get('time')
            if time != last_time:
                last_time = time
                last_time_count = 0
            else:
                last_time_count += 1

            done = terminated or truncated or (lives < 2) or (last_time_count > 10)

            # Remember
            if flag == 1 and e < 2000:
                mario.cache(state, observation, action, 0, int(done))
            else:
                mario.cache(state, observation, action, reward, int(done))

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = observation

            # Check if end of game
            if done:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=mario.exploration_rate,
                step=mario.curr_step
            )


if __name__ == '__main__':
    flag = int(sys.argv[1])
    train(flag)
