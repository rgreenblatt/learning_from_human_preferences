import collections
import itertools
from typing import Any, List, Tuple
import random

import numpy as np
from pfrl.utils.batch_states import batch_states
import torch
import pfrl
from pfrl.agents import PPO
from pfrl.agents.ppo import _mean_or_nan

from reward_model import (
    compare_via_ground_truth,
    probability_of_preferring_trajectory,
    sample_trajectory_segments_from_trajectory,
)


class Agent(PPO):
    def __init__(
        self,
        model,
        optimizer,
        reward_dataloader,
        reward_model,
        reward_opt,
        reward_update_interval,
        base_reward_proportion,
        num_envs,
        log,
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
        policy_loss_stats_window=100,
        use_reward_model=True,
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
        self.log = log
        self.trajectory_segment_len = trajectory_segment_len
        self.num_envs = num_envs
        self.use_reward_model = use_reward_model
        if self.use_reward_model:
            self._num_labels = 0
            self.reward_proportion = base_reward_proportion
            self.reward_model = reward_model
            self.reward_model.to(self.device)
            self.reward_opt = reward_opt
            self.reward_dataloader = reward_dataloader
            self.reward_update_interval = reward_update_interval
            assert_err = (
                "Currently reward_update_interval should be less than" +
                " or equal to update_interval"
            )
            assert self.reward_update_interval <= self.update_interval, assert_err

            self.rpn_loss_record = collections.deque(maxlen=100)
            self.rpn_prob_record = collections.deque(maxlen=100)

            self._reward_model_memory_ptr = 0

    def reward_train(self) -> None:
        for batch, human_labels in self.reward_dataloader:
            # uses model.zero_grad for performance reasons:
            # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
            self.reward_model.zero_grad(set_to_none=True)

            # batch: batch_dim x 2 x k x (4x84x84)
            batch = batch.to(self.device)
            human_labels = human_labels.to(self.device)
            assert batch.shape[1] == 2
            traj_1, traj_2 = batch[:, 0, ...].squeeze(), batch[:, 1, ...].squeeze()

            assert traj_1.shape == traj_2.shape
            bs, k, state_shp = traj_1.shape[0], traj_1.shape[1], traj_1.shape[2:]

            traj_1 = traj_1.reshape(bs * k, *state_shp)
            reward_sum_1 = self.reward_model(traj_1).reshape(bs, k,
                                                             1).sum(dim=1)

            traj_2 = traj_2.reshape(bs * k, *state_shp)
            reward_sum_2 = self.reward_model(traj_2).reshape(bs, k,
                                                             1).sum(dim=1)

            log_probs, probs = probability_of_preferring_trajectory(
                reward_sum_1, reward_sum_2
            )

            rpn_loss = -(log_probs * human_labels).mean()
            rpn_loss.backward()
            self.reward_opt.step()

            # TODO: add some logging

            self.rpn_loss_record.append(rpn_loss.item())
            self.rpn_prob_record.append(probs.detach().cpu().numpy())

    def _add_to_dataloader_if_dataset_is_ready(self) -> None:
        assert self.use_reward_model
        dataset_size = (
            sum(
                len(episode)
                for episode in self.memory[self._reward_model_memory_ptr:]
            ) + len(self.last_episode) + (
                0 if self.batch_last_episode is None else
                sum(len(episode) for episode in self.batch_last_episode)
            )
        )
        # TODO: consider avoiding jump cuts
        if dataset_size >= self.reward_update_interval:
            self.log.debug("Adding to dataset and running reward training")
            self._flush_last_episode()
            dataset = list(
                itertools.chain.from_iterable(
                    self.memory[self._reward_model_memory_ptr:]
                )
            )

            num_additional_labels = round(
                len(dataset) / self.trajectory_segment_len *
                self.reward_proportion
            )
            self._num_labels += num_additional_labels

            # we want to only store segments onto the cpu
            trajectory_segments, env_rews = sample_trajectory_segments_from_trajectory(
                self.trajectory_segment_len,
                num_additional_labels,
                dataset,
                lambda x: batch_states(x, torch.device("cpu"), self.phi),
                lambda x: batch_states(x, torch.device("cpu"), lambda x : x),
            )

            self.reward_dataloader.dataset.add_trajectories(
                trajectory_segments, compare_via_ground_truth(env_rews)
            )
            self.reward_train()
            self._reward_model_memory_ptr = len(self.memory)

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
                    "state": state,
                    "action": action,
                    "env_reward": reward,
                    "next_state": next_state,
                    "nonterminal": 0.0 if done else 1.0,
                }
                if self.use_reward_model:
                    with torch.no_grad():
                        transition["reward"] = self.reward_model(
                            torch.from_numpy(
                                self.phi(np.array(state))
                            ).unsqueeze(0).to(device=self.device)
                        ).item()
                else:
                    transition["reward"] = transition["env_reward"]
                self.batch_last_episode[i].append(transition)  # type: ignore
            if done or reset:
                assert self.batch_last_episode[i]  # type: ignore
                self.memory.append(self.batch_last_episode[i])  # type: ignore
                self.batch_last_episode[i] = []  # type: ignore
            self.batch_last_state[i] = None  # type: ignore
            self.batch_last_action[i] = None  # type: ignore

        if self.use_reward_model:
            self._add_to_dataloader_if_dataset_is_ready()
        self._update_if_dataset_is_ready()
        if len(self.memory) == 0:
            self._reward_model_memory_ptr = 0

    def get_extra_statistics(self) -> List[Tuple[str, Any]]:
        if self.use_reward_model:
            self.extra_stats['rpn_loss'] = _mean_or_nan(self.rpn_loss_record)
            if self.rpn_prob_record:
                rpn_probs = np.concatenate(self.rpn_prob_record,
                                           axis=0).mean(axis=0)[0]
            else:
                rpn_probs = np.nan
            self.extra_stats['rpn_probs'] = rpn_probs
            self.extra_stats['num_labels'] = self._num_labels
        return list(self.extra_stats.items())

    def get_statistics(self) -> List[Tuple[str, Any]]:
        return super().get_statistics() + self.get_extra_statistics()

    def save(self, dirname: str) -> None:
        """Save internal states."""
        super().save(dirname)

        # TODO: save and load labels!

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
