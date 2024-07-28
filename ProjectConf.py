class DefaultProjectConf:
    def __init__(self):
        # description
        self.desc = 'default configuration'

        # environment conf
        self.game_name = 'SuperMarioBros-Nes'
        self.environment_shape = 84
        self.skip_frame_num = 4
        self.render = False

        # learning conf
        self.net_name = 'MiniCnnModel'
        self.imitation = False
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.save_every = 5e5
        self.batch_size = 32
        self.gamma = 0.9
        self.min_experience_num = 1e4
        self.learn_every = 3
        self.sync_every = 1e4
        # save conf
        self.checkpoint = None

#         train conf
        self.max_episodes = 40000
        self.start_episode = 0
