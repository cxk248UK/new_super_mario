import copy
import datetime
import shutil
from pathlib import Path

import torch

from Agent import GameAgent
from environment import init_environment
from learn_log import MetricLogger
from custom_common_dict import SAVE_DIR, USE_CUDA
from ProjectConf import DefaultProjectConf
import json


def train(conf=DefaultProjectConf()):
    print(f"Using CUDA: {USE_CUDA}")

    game_env = init_environment(conf=conf)
    if conf.save_dir:
        save_dir = Path(
            conf.save_dir) / f'{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}_{game_env.unwrapped.gamename}'
    else:
        save_dir = Path(
            SAVE_DIR) / f'{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}_{game_env.unwrapped.gamename}'
    if conf.start_from_previous_result:
        save_dir = Path(conf.start_from_previous_result_save_dir)
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)

    mario = GameAgent(state_dim=(conf.skip_frame_num, conf.environment_shape, conf.environment_shape),
                      action_dim=game_env.action_space.n,
                      save_dir=save_dir, conf=conf)

    logger = MetricLogger(save_dir)

    max_total_reward = 0

    for e in range(conf.start_episode, conf.max_episodes + 1):

        state = game_env.reset()[0]

        total_reward = 0
        action_record = []

        last_time = 0
        last_time_count = 0

        # Play the game!
        while True:
            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            observation, reward, terminated, truncated, info = game_env.step(action)

            total_reward += reward
            action_record.append(action)

            lives = info.get('lives')
            time = info.get('time')
            if time != last_time:
                last_time = time
                last_time_count = 0
            else:
                last_time_count += 1

            done = terminated or truncated or (lives < 2) or (last_time_count > 10)

            # half episodes of imitation switch to classical_log soft q learning
            if mario.imitation_flag and (e == conf.imitation_episodes):
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

        if total_reward > max_total_reward:
            max_total_reward = total_reward
            max_action_record = copy.deepcopy(action_record)
            max_record = dict(max_reward=max_total_reward, action_record=max_action_record)
            torch.save(max_record, f'{save_dir}/max_record')

        if e % 1000 == 0 or e == conf.max_episodes:
            with open(f'{save_dir}/conf.json', 'w') as json_file:
                json.dump(conf.__dict__, json_file)
                json_file.close()
            # mario.memory.dumps(f'{save_dir}/memory')
            # shutil.make_archive(f'{save_dir}/memory', 'zip', f'{save_dir}/agent_last_memory')
            # shutil.rmtree(f'{save_dir}/memory')
            mario.save(Path(f'{save_dir}/last_checkpoint'))

            if conf.save_memory_1000:
                torch.save(mario.memory.sample(1000), f'{save_dir}/last_memory_1000')
