from torch import nn
import pfrl
from pfrl.policies import SoftmaxCategoricalHead


def atari_policy_value_function_model(obs_n_channels, n_actions):
    return nn.Sequential(
        nn.Conv2d(obs_n_channels, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(512, n_actions),
                SoftmaxCategoricalHead(),
            ),
            nn.Linear(512, 1),
        ),
    )


def reward_predictor_model(obs_n_channels):
    return nn.Sequential(
        nn.Conv2d(obs_n_channels, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.CELU()
    )
