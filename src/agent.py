import itertools
from typing import Any, List, Tuple
import random

import numpy as np
import torch
from torch.nn.functional import nll_loss
import pfrl
from pfrl.agents import PPO

from reward_model import compare_via_ground_truth, probability_of_preferring_trajectory, sample_trajectory_segments_from_trajectory


class Agent(PPO):
    def __init__(
        self,
        model,
        optimizer,
        reward_dataloader,
        reward_model,
        reward_opt,
        reward_update_interval,
        num_envs,
        trajectory_segment_len=25,
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
        batch_states=pfrl.utils.batch_states,
        recurrent=False,
        max_recurrent_sequence_len=None,
        act_deterministically=False,
        max_grad_norm=None,
        value_stats_window=1000,
        entropy_stats_window=1000,
        value_loss_stats_window=100,
        policy_loss_stats_window=100
    ):
        super().__init__(
            model,
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
            policy_loss_stats_window=policy_loss_stats_window
        )
        if self.recurrent:
            raise ValueError("This does not support recurrent nets")
        self.extra_stats = {}
        self.t = 0
        self.trajectory_segment_len = trajectory_segment_len
        self.reward_proportion = 1
        self.num_envs = num_envs
        self.reward_model = reward_model
        self.reward_opt = reward_opt
        self.reward_dataloader = reward_dataloader
        self.reward_update_interval = reward_update_interval
        assert self.reward_update_interval < self.update_interval, "Currently reward_update_interval should be less than update_interval"

    def reward_training_loop(self) -> None:
        # uses model.zero_grad for performance reasons:
        # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
        self.reward_model.zero_grad(set_to_none=True)
        for batch, human_labels in self.reward_dataloader:
            # batch: batch_dim x 2 x k x (84x84x4)
            assert batch.shape[1] == 2
            traj_1, traj_2 = batch[:, 0, ...].squeeze(), batch[:, 1, ...].squeeze()
            reward_sum_1 = self.reward_model(traj_1).sum(dim=3)
            reward_sum_2 = self.reward_model(traj_2).sum(dim=3)
            log_probs, probs = probability_of_preferring_trajectory(
                reward_sum_1, reward_sum_2
            )
            reward_loss = nll_loss(log_probs, human_labels, reduction='mean')
            reward_loss.backward()
            # TODO: add some logging
            self.extra_stats['rpn_loss'] = reward_loss.item()
            self.extra_stats['rpn_probs'] = probs

    def _add_to_dataloader_if_dataset_is_ready(self, episodes) -> None:
        dataset_size = (
            sum(len(episode) for episode in episodes) + len(self.last_episode) +
            (
                0 if self.batch_last_episode is None else
                sum(len(episode) for episode in self.batch_last_episode)
            )
        )
        if dataset_size >= self.reward_update_interval:
            states = [
                b['state'] for b in itertools.chain.from_iterable(episodes)
            ]
            trajectory_segments, env_rews = sample_trajectory_segments_from_trajectory(
                self.trajectory_segment_len,
                int(self.t * self.reward_proportion / self.num_envs),
                states
            )

            self.reward_dataloader.add_trajectories(
                trajectory_segments, compare_via_ground_truth(env_rews)
            )

    def _batch_observe_train(
        self, batch_obs, batch_reward, batch_done, batch_reset
    ):
        assert self.training

        for i, (state, action, reward, next_state, done, reset) in enumerate(
            zip(
                self.batch_last_state,
                self.batch_last_action,
                batch_reward,
                batch_obs,
                batch_done,
                batch_reset
            )  # type: ignore
        ):
            if state is not None:
                assert action is not None
                transition = {
                    "state":
                        state,
                    "action":
                        action,
                    "env_reward":
                        reward,
                    "reward":
                        self.reward_model(
                            torch.from_numpy(self.phi(np.array(state))
                                            ).unsqueeze(0)
                        ).item(),
                    "next_state":
                        next_state,
                    "nonterminal":
                        0.0 if done else 1.0,
                }
                self.batch_last_episode[i].append(transition)  # type: ignore
            if done or reset:
                assert self.batch_last_episode[i]  # type: ignore
                self.memory.append(self.batch_last_episode[i])  # type: ignore
                self.batch_last_episode[i] = []  # type: ignore
            self.batch_last_state[i] = None  # type: ignore
            self.batch_last_action[i] = None  # type: ignore

        self._add_to_dataloader_if_dataset_is_ready(self.memory)
        self._update_if_dataset_is_ready()

    def get_extra_statistics(self) -> List[Tuple[str, Any]]:
        return list(self.extra_stats.items())

    def get_statistics(self) -> List[Tuple[str, Any]]:
        return super().get_statistics() + self.get_extra_statistics()

    def save(self, dirname: str) -> None:
        """Save internal states."""
        super().save(dirname)
        print(f"saving with t: {self.t}")

        torch.save(torch.get_rng_state(), f'{dirname}/torch_rng_state.pt')
        torch.save(np.random.get_state(), f'{dirname}/numpy_rng_state.pt')
        torch.save(random.getstate(), f'{dirname}/python_rng_state.pt')

        torch.save(self.t, f'{dirname}/time_step.pt')
        torch.save(self.n_updates, f'{dirname}/n_updates.pt')

    def load(self, dirname: str) -> None:
        """Load internal states."""
        super().load(dirname)

        torch.set_rng_state(torch.load(f'{dirname}/torch_rng_state.pt'))
        np.random.set_state(torch.load(f'{dirname}/numpy_rng_state.pt'))
        random.setstate(torch.load(f'{dirname}/python_rng_state.pt'))

        self.t = torch.load(f'{dirname}/time_step.pt')
        self.n_updates = torch.load(f'{dirname}/n_updates.pt')
