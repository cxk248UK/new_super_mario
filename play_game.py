import json

from environment import init_environment
from gymnasium.wrappers import RecordVideo


def play():
    game_env = init_environment(render=True)
    game_env = RecordVideo(game_env, video_folder='video_record')

    state = game_env.reset()[0]

    action_record_action = json.load(open('max_reward_actions.json', 'r'))

    for action in action_record_action:
        # Run agent on the state

        # Agent performs action
        observation, reward, terminated, truncated, info = game_env.step(action)

        game_env.render()

        # Update state
        state = observation

        # Check if end of game
        if terminated or truncated:
            break

    game_env.close()


if __name__ == '__main__':
    play()
