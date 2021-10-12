import gym


class RewardModelWrapper(gym.RewardWrapper):
    def __init__(self, env, sample_prop):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        # TODO
        raise NotImplementedError
