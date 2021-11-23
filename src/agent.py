import collections
import itertools
from typing import Any, List, Tuple
import random
from time import sleep

import numpy as np
from pfrl.utils.batch_states import batch_states
import torch
import pfrl
from pfrl.agents import PPO
from pfrl.agents.ppo import _mean_or_nan

from reward_model import (
    probability_of_preferring_trajectory,
    sample_trajectory_segments_from_trajectory,
)
from comparison_collectors import (
    SyntheticComparisonCollector, HumanComparisonCollector
)


class Agent(PPO):
    def __init__(
        self,
        model,
        optimizer,
        env,
        reward_dataloader,
        reward_model,
        reward_opt,
        reward_update_interval,
        reward_episode_chopping_interval,
        ground_truth_human_like,
        base_reward_proportion,
        rpn_num_full_prop_updates,
        reward_inference_batch_size,
        num_envs,
        log,
        experiment_name,
        human_labels=False,
        human_error_rate=0.1,
        trajectory_segment_len=32,
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
        self.num_envs = num_envs
        self.use_reward_model = use_reward_model
        if self.use_reward_model:
            self._n_rpn_updates = 0
            self._num_labels = 0
            self.trajectory_segment_len = trajectory_segment_len
            self.reward_proportion = base_reward_proportion
            self._rpn_num_full_prop_updates = rpn_num_full_prop_updates
            self.reward_inference_batch_size = reward_inference_batch_size
            self.reward_model = reward_model
            self.reward_model.to(self.device)
            self.reward_opt = reward_opt
            self.reward_dataloader = reward_dataloader
            self.reward_update_interval = reward_update_interval
            self.reward_episode_chopping_interval = reward_episode_chopping_interval

            self._reward_model_dataset_memory = []
            self._reward_model_batch_last_episode = None

            self.rpn_loss_record = collections.deque(maxlen=100)
            self.rpn_prob_record = collections.deque(maxlen=100)

            self._collector = (
                HumanComparisonCollector(
                    experiment_name, env, human_error_rate=human_error_rate
                ) if human_labels else
                SyntheticComparisonCollector(ground_truth_human_like)
            )

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

    # returns true if updated this iteration
    def _add_to_dataloader_if_dataset_is_ready(self) -> None:
        assert self.use_reward_model
        dataset_size = sum(
            len(episode) for episode in self._reward_model_dataset_memory
        )

        if dataset_size >= self.reward_update_interval:
            dataset = list(
                itertools.chain.from_iterable(
                    self._reward_model_dataset_memory
                )
            )
            if self._n_rpn_updates < self._rpn_num_full_prop_updates:
                reward_proportion = 1.0
            else:
                reward_proportion = self.reward_proportion
            num_additional_labels = round(
                len(dataset) / self.trajectory_segment_len * reward_proportion
            )
            self._num_labels += num_additional_labels

            # we want to only store segments onto the cpu
            # TODO: async approach and uncertainty based sampling
            trajectory_segments, env_rews, times = sample_trajectory_segments_from_trajectory(
                self.trajectory_segment_len,
                num_additional_labels,
                dataset,
                lambda x: batch_states(x, torch.device("cpu"), self.phi),
                lambda x: batch_states(x, torch.device("cpu"), lambda x : x),
            )
            human_obs = None # TODO
            self._collector.add_values(trajectory_segments, env_rews, times, human_obs)

            while len(self._collector) < self._collector.n_labeled_comparisons():
                print("waiting for labels!")
                sleep(5)

            trajectories, mus, times = self._collector.pop_labeled()

            self.reward_dataloader.dataset.add_trajectories(
                trajectories, mus, times,
            )
            n_epochs = round(dataset_size / self.reward_update_interval)
            if self._n_rpn_updates == self._rpn_num_full_prop_updates - 1:
                extra_epochs = 50
                self.log.info(
                    f"running {extra_epochs} extra reward model training epochs with"
                    " initial data"
                )
                n_epochs += extra_epochs

            self.log.debug(
                "running {} reward model training of len {} (batches={})".
                format(
                    n_epochs,
                    len(self.reward_dataloader.dataset),
                    len(self.reward_dataloader)
                )
            )
            for _ in range(n_epochs):
                self.reward_train()

            self._reward_model_dataset_memory = []
            self._n_rpn_updates += 1

    def _batch_observe_train(
        self, batch_obs, batch_reward, batch_done, batch_reset
    ):
        assert self.training
        assert self.batch_last_episode is not None
        assert self.batch_last_state is not None
        assert self.batch_last_action is not None

        if self._reward_model_batch_last_episode is None:
            self._reward_model_batch_last_episode = [
                [] for _ in self.batch_last_episode
            ]

        for i, (state, action, reward, next_state, done, reset) in enumerate(
            zip(
                self.batch_last_state,
                self.batch_last_action,
                batch_reward,
                batch_obs,
                batch_done,
                batch_reset
            )
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
                if not self.use_reward_model:
                    transition["reward"] = transition["env_reward"]
                # otherwise, rewards are handled in a batch for efficiency

                self.batch_last_episode[i].append(transition)
                self._reward_model_batch_last_episode[i].append(
                    (transition, self.t)
                )

            def move_to_memory(batch_episodes, memory):
                assert batch_episodes[i]
                memory.append(batch_episodes[i])
                batch_episodes[i] = []

            print("T:", self.t)

            if done or reset:
                move_to_memory(self.batch_last_episode, self.memory)
                move_to_memory(
                    self._reward_model_batch_last_episode,
                    self._reward_model_dataset_memory
                )

            elif len(
                self._reward_model_batch_last_episode[i]
            ) >= self.reward_episode_chopping_interval:
                move_to_memory(
                    self._reward_model_batch_last_episode,
                    self._reward_model_dataset_memory
                )

            self.batch_last_state[i] = None
            self.batch_last_action[i] = None

        if self.use_reward_model:
            self._add_to_dataloader_if_dataset_is_ready()
        self._update_if_dataset_is_ready()
        if len(self.memory) == 0:
            self._reward_model_memory_ptr = 0

    def _flush_last_episode(self):
        if self.use_reward_model:

            def all_episodes():
                if self.last_episode:
                    yield self.last_episode
                if self.batch_last_episode:
                    for episode in self.batch_last_episode:
                        yield episode
                for episode in self.memory:
                    yield episode

            to_update = []
            for episode in all_episodes():
                for transition in episode:
                    if "reward" in transition:
                        return
                    to_update.append(transition)

            for batch_start in range(
                0, len(to_update), self.reward_inference_batch_size
            ):
                self._set_batch_rewards(
                    to_update[batch_start:batch_start +
                              self.reward_inference_batch_size]
                )

        super()._flush_last_episode()

    @torch.no_grad()
    def _set_batch_rewards(self, batch_subset):
        states = torch.stack(
            [
                torch.from_numpy(self.phi(np.array(transition["state"])))
                for transition in batch_subset
            ],
            dim=0,
        ).to(device=self.device)
        rewards = self.reward_model(states).cpu()
        assert len(rewards) == len(batch_subset)

        for transition, reward in zip(batch_subset, rewards):
            transition["reward"] = reward.item()

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
            self.extra_stats['n_rpn_updates'] = self._n_rpn_updates
            self.extra_stats[ 'current_num_labels' ] = \
                self.reward_dataloader.dataset.current_num_labels()
            self.extra_stats['rpn_dataset_len'] = \
                len(self.reward_dataloader.dataset)
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
