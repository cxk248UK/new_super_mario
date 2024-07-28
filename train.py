import datetime
from pathlib import Path

from Agent import GameAgent
from environment import init_environment
from learn_log import MetricLogger
from custom_common_dict import SAVE_DIR, USE_CUDA
from ProjectConf import DefaultProjectConf
import json


def train(conf=DefaultProjectConf()):
    print(f"Using CUDA: {USE_CUDA}")

    game_env = init_environment(conf=conf)
    save_dir = Path(SAVE_DIR) / f'{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}_{game_env.unwrapped.gamename}'
    save_dir.mkdir(parents=True)

    mario = GameAgent(state_dim=(conf.skip_frame_num, conf.environment_shape, conf.environment_shape),
                      action_dim=game_env.action_space.n,
                      save_dir=save_dir, conf=conf)

    logger = MetricLogger(save_dir)

    for e in range(conf.start_episode, conf.max_episodes):

        state = game_env.reset()[0]

        last_time = 0
        last_time_count = 0

        # Play the game!
        while True:
            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            observation, reward, terminated, truncated, info = game_env.step(action)
            lives = info.get('lives')
            time = info.get('time')
            if time != last_time:
                last_time = time
                last_time_count = 0
            else:
                last_time_count += 1

            done = terminated or truncated or (lives < 2) or (last_time_count > 10)

            # half episodes of imitation switch to classical soft q learning
            if mario.imitation_flag and (e > (conf.max_episodes / 2)):
                mario.switch_imitation()

            # Remember
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

        if e % 1000 == 0 or e == conf.max_episodes-1:
            with open(f'{save_dir}/conf.json', 'w') as json_file:
                json.dump(conf.__dict__, json_file)
                json_file.close()
