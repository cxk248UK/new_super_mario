import gymnasium as gym
import numpy as np
import retro
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, TransformObservation
import torch
from custom_common_dict import ALLOW_ACTION_COMBOS, OBSERVATION_SHAP, GAME_NAME, TRAIN_RENDER, PLAY_RENDER
from ProjectConf import DefaultProjectConf


# 动作包装
class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


# 设定允许的动作组合
class SonicDiscretizer(Discretizer):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(
            env=env,
            combos=ALLOW_ACTION_COMBOS,
        )


# 将多个步骤合并
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        observation, reward, terminated, truncated, info = None, None, None, None, None
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, info


def init_environment(render=False, conf=DefaultProjectConf()):
    if render:
        game_env = retro.make(game=conf.game_name, render_mode=PLAY_RENDER)
    else:
        game_env = retro.make(game=conf.game_name, render_mode=TRAIN_RENDER)

    game_env = SonicDiscretizer(game_env)

    # Apply Wrappers to environment
    game_env = SkipFrame(game_env, skip=conf.skip_frame_num)
    game_env = GrayScaleObservation(game_env, keep_dim=False)
    game_env = ResizeObservation(game_env, shape=conf.environment_shape)
    game_env = TransformObservation(game_env, f=lambda x: x / 255.)
    game_env = FrameStack(game_env, num_stack=conf.skip_frame_num)
    return game_env
