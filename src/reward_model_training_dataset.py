from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RewardModelTrainingDataset(Dataset):
    """
    Dataset which allows for sliding window of states and preferences.

    We sample older trajectories exponentially less frequently. Precisely, a
    group of trajectories added i iterations ago is sampled with probability
    proportional to sample_decay**i.

    To save on memory, if the probability of sampling from a group of
    trajectories is less than min_sample_prob, that group is eliminated.

    The size is arbitrary because we always sample randomly, but
    it is computed as size_multiplier * n / p
    Where:
        p: probability of sampling from most recent group
        n: number of trajectories in most recent group

    Args:
        sample_decay (float): see above
        min_sample_prob (float): see above
        epoch_size_multiplier (float): see above
    """
    def __init__(
        self,
        sample_decay: float = 0.5,
        min_sample_prob: float = 1e-4,
        size_multiplier: float = 1.5,
        human_error_rate: float = 0.1
    ) -> None:
        super().__init__()

        assert 0. < sample_decay < 1.
        assert 0. <= min_sample_prob < 1.
        assert size_multiplier > 0.
        assert 0. <= human_error_rate < 0.5

        self._sample_decay = sample_decay
        self._min_sample_prob = min_sample_prob
        self._size_multiplier = size_multiplier

        self._samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._probs: List[float] = []

        self._idx_0_pref = torch.tensor(
            [1 - human_error_rate, human_error_rate]
        )
        self._idx_1_pref = torch.tensor(
            [human_error_rate, 1 - human_error_rate]
        )

    def add_trajectories(
        self, states: torch.Tensor, preferences: torch.Tensor
    ) -> None:
        """
        Args:
            states (Tensor): [n x 2 x k x (state dim)]
            preferences (Tensor): [n] bool, true if idx 0 is prefered over idx 1

        Where n is number of trajectories, k is trajectory length, and the
        state dim is whatever remaining dimensions the state has.
        """
        assert states.size(0) == preferences.size(0)
        assert states.size(1) == 2

        self._samples.append((states, preferences))
        self._compute_probs()

        if self._probs[0] < self._min_sample_prob:
            del self._samples[0]
            self._compute_probs()

    def _raw_proportional_constant(self, idx: int) -> float:
        return self._sample_decay**(-idx)

    def _compute_probs(self):
        norm_const = sum(
            self._raw_proportional_constant(idx)
            for idx in range(len(self._samples))
        )
        self._probs = [
            self._raw_proportional_constant(idx) / norm_const
            for idx in range(len(self._samples))
        ]

    def __len__(self) -> int:
        if len(self._samples) == 0:
            return 0
        return round(
            self._size_multiplier * self._samples[-1][0].size(0) /
            self._probs[-1]
        )

    # NOTE: idx input isn't relevant, a bit gross...
    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: consider instead doing this with numpy generator object and
        # accepting seed as arg
        group_idx = np.random.choice(len(self._samples), p=self._probs)
        n = self._samples[group_idx][0].size(0)
        idx = np.random.randint(0, n)

        x = self._samples[group_idx][0][idx]

        if bool(self._samples[group_idx][1][idx]):
            label_probs = self._idx_0_pref
        else:
            label_probs = self._idx_1_pref

        return x, label_probs


if __name__ == "__main__":
    dataset = RewardModelTrainingDataset(min_sample_prob=5e-3)

    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0
    )

    print("DataLoader tests")
    for batched in dataloader:
        print(batched.shape)
        print(batched[0].size(), batched[1].size())

    print("adding trajs tests")
    for _ in range(10):
        dataset.add_trajectories(
            torch.rand(100, 2, 30), torch.full((100,), False)
        )
        print(dataset._probs)
        print(len(dataset))

    # sampling??
    for _ in range(100):
        dataset[0]

    print("sampling after adding tests")
    for batched in dataloader:
        print(batched[0].size(), batched[1].size())
