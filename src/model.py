from torch import nn
import pfrl
from pfrl.policies import SoftmaxCategoricalHead


def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer


def atari_policy_value_function_model(obs_n_channels, n_actions):
    # TODO:
    # - Should we tweak model at all?
    # - Where is this model architecture from?
    # - Why don't we just use default init?
    return nn.Sequential(
        lecun_init(nn.Conv2d(obs_n_channels, 32, 8, stride=4)),
        nn.ReLU(),
        lecun_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        lecun_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        lecun_init(nn.Linear(3136, 512)),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                lecun_init(nn.Linear(512, n_actions), 1e-2),
                SoftmaxCategoricalHead(),
            ),
            lecun_init(nn.Linear(512, 1)),
        ),
    )
