import functools

import numpy as np
import torch
import gym
import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO
from pfrl.wrappers import atari_wrappers

from model import atari_policy_value_function_model


# TODO: consider splitting this up further

class RewardModelWrapper(gym.RewardWrapper):
    def __init__(self, env, sample_prop):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        # TODO
        raise NotImplementedError


def make_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        type=str,
                        default="BreakoutNoFrameskip-v4",
                        help="Gym Env ID.")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="GPU device ID. Set to -1 to use CPUs only.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of env instances run in parallel.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=("Directory path to save output files."
              " If it does not exist, it will be created."),
    )
    parser.add_argument("--steps",
                        type=int,
                        default=10**7,
                        help="Total time steps for training.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--lr",
                        type=float,
                        default=2.5e-4,
                        help="Learning rate.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100000,
        help="Interval (in timesteps) between evaluation phases.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
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
        help=("Directory path to load a saved agent data from"
              " if it is a non-empty string."),
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
        help=
        ("Monitor env. Videos and additional information are saved as output files."
         ),
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=128 * 8,
        help="Interval (in timesteps) between PPO iterations.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=32 * 8,
        help="Size of minibatch (in timesteps).",
    )
    parser.add_argument(
        "--reward-model-sample-prop",
        type=float,
        default=None,
        help="proportion of time steps used for reward model training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs used for each PPO iteration.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10000,
        help="Interval (in timesteps) of printing logs.",
    )
    parser.add_argument(
        "--no-frame-stack",
        action="store_true",
        default=False,
        help=
        ("Disable frame stacking so that the agent can only see the current screen."
         ),
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="Frequency at which agents are stored.",
    )

    return parser


def main():
    args = make_parser().parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2**31

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2**32 - 1 - process_seed if test else process_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test,
            flicker=False,
            frame_stack=False,
        )
        if args.reward_model_sample_prop is not None:
            env = RewardModelWrapper(env, args.reward_model_sample_prop)
        env.seed(env_seed)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training")
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        vec_env = pfrl.envs.MultiprocessVectorEnv([
            functools.partial(make_env, idx, test)
            for idx in range(args.num_envs)
        ])
        if not args.no_frame_stack:
            vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
        return vec_env

    sample_env = make_batch_env(test=False)
    print("Observation space", sample_env.observation_space)
    print("Action space", sample_env.action_space)
    n_actions = sample_env.action_space.n
    obs_n_channels = sample_env.observation_space.low.shape[0]
    del sample_env

    model = atari_policy_value_function_model(obs_n_channels, n_actions)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # TODO: add reward model and reward model optimizer (also use adam, use
    # different lr, add use some weight decay)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = PPO(
        model,
        opt,
        gpu=args.gpu,
        phi=phi,
        update_interval=args.update_interval,
        minibatch_size=args.batchsize,
        epochs=args.epochs,
        clip_eps=0.1,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=1e-2,
        recurrent=False,
        max_grad_norm=0.5,
    )
    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
        )
        print("n_runs: {} mean: {} median: {} stdev: {}".format(
            args.eval_n_runs,
            eval_stats["mean"],
            eval_stats["median"],
            eval_stats["stdev"],
        ))
    else:
        step_hooks = []

        # Linearly decay the learning rate to zero
        # TODO: consider changing this to something more sensible
        def lr_setter(_, agent, value):
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value

        step_hooks.append(
            experiments.LinearInterpolationHook(args.steps, args.lr, 0,
                                                lr_setter))

        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            checkpoint_freq=args.checkpoint_frequency,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            save_best_so_far_agent=False,
            step_hooks=step_hooks,
        )


if __name__ == "__main__":
    main()
