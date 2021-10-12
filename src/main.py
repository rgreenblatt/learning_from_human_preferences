import functools

import numpy as np
import torch
import gym
import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO
from pfrl.wrappers import atari_wrappers

from model import atari_policy_value_function_model
from arg_parser import make_parser

# TODO: consider splitting this up further


class RewardModelWrapper(gym.RewardWrapper):
    def __init__(self, env, sample_prop):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        # TODO
        raise NotImplementedError


class PPOExtraStats(PPO):
    def __init__(self,
                 model,
                 optimizer,
                 obs_normalizer=None,
                 gpu=None,
                 gamma=0.99,
                 lambd=0.95,
                 phi=lambda x: x,
                 value_func_coef=1,
                 entropy_coef=0.01,
                 update_interval=2048,
                 minibatch_size=64,
                 epochs=10,
                 clip_eps=0.2,
                 clip_eps_vf=None,
                 standardize_advantages=True,
                 batch_states=...,
                 recurrent=False,
                 max_recurrent_sequence_len=None,
                 act_deterministically=False,
                 max_grad_norm=None,
                 value_stats_window=1000,
                 entropy_stats_window=1000,
                 value_loss_stats_window=100,
                 policy_loss_stats_window=100):
        super().__init__(model,
                         optimizer,
                         obs_normalizer=obs_normalizer,
                         gpu=gpu,
                         gamma=gamma,
                         lambd=lambd,
                         phi=phi,
                         value_func_coef=value_func_coef,
                         entropy_coef=entropy_coef,
                         update_interval=update_interval,
                         minibatch_size=minibatch_size,
                         epochs=epochs,
                         clip_eps=clip_eps,
                         clip_eps_vf=clip_eps_vf,
                         standardize_advantages=standardize_advantages,
                         batch_states=batch_states,
                         recurrent=recurrent,
                         max_recurrent_sequence_len=max_recurrent_sequence_len,
                         act_deterministically=act_deterministically,
                         max_grad_norm=max_grad_norm,
                         value_stats_window=value_stats_window,
                         entropy_stats_window=entropy_stats_window,
                         value_loss_stats_window=value_loss_stats_window,
                         policy_loss_stats_window=policy_loss_stats_window)
        self.extra_stats = {}

    def get_extra_statistics(self):
        return list(self.extra_stats.items())

    def get_statistics(self):
        return super().get_statistics() + self.get_extra_statistics()


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

    args.outdir = experiments.prepare_output_dir(args,
                                                 args.outdir,
                                                 exp_id=args.exp_id)
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

    agent = PPOExtraStats(
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

        agent.extra_stats["lr"] = args.lr

        # Linearly decay the learning rate to zero
        # TODO: consider changing this to something more sensible
        def lr_setter(_, agent, value):
            agent.extra_stats["lr"] = value
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
            use_tensorboard=True,
        )


if __name__ == "__main__":
    main()
