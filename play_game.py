from environment import init_environment
from custom_common_dict import TRAINED_MODEL_PATH, USE_CUDA, FRAME_SKIP, FRAME_WIDTH, FRAME_HIGH
from agent import GameAgent


def play():
    game_env = init_environment(render=True)

    state = game_env.reset()[0]

    mario = GameAgent(state_dim=(FRAME_SKIP, FRAME_WIDTH, FRAME_HIGH), action_dim=game_env.action_space.n,
                      save_dir=None,
                      checkpoint=TRAINED_MODEL_PATH)

    while True:
        # Run agent on the state
        action = mario.act(state, True)

        # Agent performs action
        observation, reward, terminated, truncated, info = game_env.step(action)

        game_env.render()

        # Update state
        state = observation

        # Check if end of game
        if terminated or truncated:
            break


if __name__ == '__main__':
    play()
