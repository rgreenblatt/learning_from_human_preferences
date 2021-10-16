import argparse

from utils import float2int


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-id",
        type=str,
        default=None,
        help="experiment id used from results directory (should be unique)"
    )
    parser.add_argument(
        "--env", type=str, default="BreakoutNoFrameskip-v4", help="Gym Env ID."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Set to -1 to use CPUs only."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of env instances run in parallel.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed [0, 2 ** 32)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10**7,
        help="Total time steps for training."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument(
        "--lr", type=float, default=2.5e-4, help="Learning rate."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0, help="Adam weight decay."
    )
    parser.add_argument(
        "--reward-model-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the reward model."
    )
    parser.add_argument(
        "--reward-model-weight-decay",
        type=float,
        default=1e-5,
        help="Adam weight decay for the reward model."
    )
    parser.add_argument(
        "--eval-interval",
        type=float2int,
        default=100000,
        help="Interval (in timesteps) between evaluation phases.",
    )
    parser.add_argument(
        "--base-reward-proportion",
        type=float,
        default=0.5,
        help="initial reward proportion (decays)",
    )
    parser.add_argument(
        "--reward-model-training-size-multiplier",
        type=float,
        default=3.,
        help="determines amount of data trained on per each time data "
        "is added for the reward model",
    )
    parser.add_argument(
        "--reward-proportion-update-freq",
        type=float2int,
        default=5e6,
        help="Interval (in timesteps) between evaluation phases.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=1,
        help="Number of episodes ran in an evaluation phase.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run demo episodes, not training.",
    )
    # add some args for visualizing trained reward model vs actual reward
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help=(
            "Directory path to load a saved agent data from"
            " if it is a non-empty string."
        ),
    )
    parser.add_argument(
        "--no-reward-model",
        dest="use_reward_model",
        action="store_false",
        help="Don't use a reward model",
    )
    parser.add_argument(
        "--use-reward-model",
        dest="use_reward_model",
        action="store_true",
        help="Use a reward model",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--update-interval",
        type=float2int,
        default=128 * 8,
        help="Interval (in timesteps) between PPO iterations.",
    )
    parser.add_argument(
        "--batchsize",
        type=float2int,
        default=32 * 8,
        help="Size of minibatch (in timesteps).",
    )
    parser.add_argument(
        "--rpn-batchsize",
        type=float2int,
        default=32 * 8,
        help="Size of minibatch (in timesteps).",
    )
    parser.add_argument(
        "--rpn-sample-prop",
        type=float,
        default=0.25,
        help="proportion of update_interval timesteps used for rpn training",
    )
    parser.add_argument(
        "--reward-model-sample-decay",
        type=float,
        default=0.9,
        help="decay per group of reward model samples",
    )
    parser.add_argument(
        "--min-sample-prob",
        type=float,
        default=5e-3,
        help="minimum sample probability for reward dataset",
    )
    parser.add_argument(
        "--epochs",
        type=float2int,
        default=4,
        help="Number of epochs used for each PPO iteration.",
    )
    parser.add_argument(
        "--log-interval",
        type=float2int,
        default=1e3,
        help="Interval (in timesteps) of printing logs.",
    )
    parser.add_argument(
        "--no-frame-stack",
        action="store_true",
        default=False,
        help=(
            "Disable frame stacking so that the agent can only see the current screen."
        ),
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=float2int,
        default=None,
        help="Frequency at which agents are stored.",
    )

    return parser
