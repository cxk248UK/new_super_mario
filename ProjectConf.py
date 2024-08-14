class DefaultProjectConf:
    def __init__(self):
        # description
        self.desc = 'default configuration'
        self.save_dir = 'default_save_dir'

        # environment conf
        self.game_name = 'SuperMarioBros-Nes'
        self.environment_shape = 84
        self.skip_frame_num = 4
        self.render = False

        # learning conf
        self.net_name = 'MiniCnnModel'
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.save_every = 5e5
        self.batch_size = 32
        self.gamma = 0.9
        self.min_experience_num = 1e4
        self.learn_every = 3
        self.sync_every = 1e4
        self.save_memory_to_memory = False
        self.save_memory_1000 = False

        # imitation conf
        self.imitation = False
        self.imitation_decay = 0
        self.imitation_episodes = 10000
        self.imitation_exploration_rate = 1.0
        self.imitation_exploration_rate_decay = 0.99999975

        # save conf
        self.checkpoint = None

        # train conf
        self.max_episodes = 40000
        self.start_episode = 0

        # continue learning
        self.start_from_previous_result = False
        self.start_from_previous_result_log = None
        self.start_from_previous_result_save_dir = None
        self.start_from_previous_result_save_model = None

        self.is_colab = False
