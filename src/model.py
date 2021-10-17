import torch
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


class AddX(nn.Module):
    def __init__(self, adder):
        super().__init__()
        self.register_buffer('adder', torch.tensor(float(adder)))

    def forward(self, x):
        return x + self.adder


def modify_bias_init(layer: nn.Linear, bias: float) -> nn.Linear:
    nn.init.constant_(layer.bias, bias)
    return layer


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
        modify_bias_init(nn.Linear(64, 1), -4.0),
        nn.CELU(alpha=1.0),
        AddX(1.0),
    )
