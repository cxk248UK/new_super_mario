import gymnasium as gym
import numpy as np
import retro
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

from custom_common_dict import ALLOW_ACTION_COMBOS, OBSERVATION_SHAP


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
        return observation, reward, terminated, truncated, info

# game_env = retro.make(game="SuperMarioBros-Nes", render_mode='human')

game_env = retro.make(game="SuperMarioBros-Nes", render_mode='None')

game_env = SonicDiscretizer(game_env)

# 每四帧合并为一个帧
game_env = SkipFrame(game_env, 4)

# RGB转换为灰度并减少一个颜色维度
game_env = GrayScaleObservation(game_env, keep_dim=False)

# 间原观察空间降采样为正方形
game_env = ResizeObservation(game_env, OBSERVATION_SHAP)

# 将关联的四帧合并到一个观察状态中
game_env = FrameStack(game_env, num_stack=4)
