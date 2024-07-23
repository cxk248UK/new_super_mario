import datetime
from pathlib import Path

from environment import init_environment
from learn_log import MetricLogger
from org_agent import Mario

# Initialize Super Mario environment
env = init_environment(render=False)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None  # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 40000

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = env.reset()[0]

    # Play the game!
    while True:

        # 3. Show environment (the visual) [WIP]
        # env.render()

        # 4. Run agent on the state
        action = mario.act(state)

        # 5. Agent performs action
        next_state, reward, terminated, truncated, info = env.step(action)

        # 6. Remember
        mario.cache(state, next_state, action, reward, int(terminated or truncated))

        # 7. Learn
        q, loss = mario.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if terminated or truncated:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
