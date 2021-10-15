import functools

import numpy as np
import torch
import pfrl
from pfrl import experiments, utils
from pfrl.wrappers import atari_wrappers
from torch.utils.data.dataloader import DataLoader

from model import atari_policy_value_function_model, reward_predictor_model
from arg_parser import make_parser
from reward_model_training_dataset import RewardModelTrainingDataset
from utils import PrintAndLogStdoutStderr
from agent import Agent


def main():
    args = make_parser().parse_args()

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2**31

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, exp_id=args.exp_id
    )
    output_logger = PrintAndLogStdoutStderr(args.outdir)

    import logging
    logging.basicConfig(level=args.log_level)

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
        env.seed(env_seed)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        vec_env = pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx in range(args.num_envs)
            ]
        )
        if not args.no_frame_stack:
            vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
        return vec_env

    sample_env = make_batch_env(test=False)
    print("Observation space", sample_env.observation_space.shape)
    print("Action space", sample_env.action_space.shape)
    n_actions = sample_env.action_space.n
    obs_n_channels = sample_env.observation_space.low.shape[0]
    del sample_env

    model = atari_policy_value_function_model(obs_n_channels, n_actions)
    reward_model = reward_predictor_model(obs_n_channels)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        eps=1e-5,
        weight_decay=args.weight_decay
    )
    reward_opt = torch.optim.Adam(
        reward_model.parameters(),
        lr=args.lr,
        eps=1e-5,
        weight_decay=args.weight_decay
    )

    # TODO: add reward model and reward model optimizer (also use adam, use
    # different lr, add use some weight decay)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    dataset = RewardModelTrainingDataset(min_sample_prob=args.min_sample_prob)
    reward_dataloader = DataLoader(
        dataset, batch_size=args.rpn_batchsize, shuffle=False, num_workers=0
    )

    agent = Agent(
        model,
        opt,
        reward_model=reward_model,
        reward_opt=reward_opt,
        reward_update_interval=int(args.update_interval * args.rpn_sample_prop),
        num_envs=args.num_envs,
        reward_dataloader=reward_dataloader,
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
        print(f'loaded with timestep {agent.t}')

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        step_hooks = []

        agent.extra_stats["lr"] = args.lr

        # Linearly decay the learning rate to zero
        # TODO: consider changing this to something more sensible
        def lr_setter(_, agent, value):
            agent.extra_stats["lr"] = value
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value

        step_hooks.append(
            experiments.LinearInterpolationHook(
                args.steps, args.lr, 0, lr_setter
            )
        )

        def update_time_hook(_, agent, t):
            agent.t = t

        def update_reward_prop(_, agent, t):
            if t > 0 and t % args.reward_proportion_update_freq == 0:
                agent.reward_proportion = args.reward_proportion_update_freq / (
                    t + args.reward_proportion_update_freq
                )

        def run_reward_training(_, agent, t):
            if t > 0 and t % int(
                args.rpn_sample_prop * args.update_interval
            ) == 0:
                agent.reward_training_loop()

        step_hooks.append(update_time_hook)
        step_hooks.append(update_reward_prop)
        step_hooks.append(run_reward_training)

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
            use_tensorboard=True,
            step_offset=agent.t,
        )

    del output_logger


if __name__ == "__main__":
    main()
