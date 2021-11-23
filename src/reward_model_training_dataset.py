from typing import List, Optional, Tuple

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
    it is computed as round((size_multiplier * n / p) / batch_size) * batch_size
    Dividing and multiplying by batch_size ensures the size is a multiple
    of batch size. This is only done if batch size isn't none.
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
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert 0. < sample_decay < 1.
        assert 0. <= min_sample_prob < 1.
        assert size_multiplier > 0.
        assert batch_size is None or batch_size > 0

        self._sample_decay = sample_decay
        self._min_sample_prob = min_sample_prob
        self._size_multiplier = size_multiplier
        self._batch_size = batch_size

        self._states : List[torch.Tensor] = []
        self._mus : List[torch.Tensor] = []
        self._times : List[Tuple[int, int]] = []
        self._probs: List[float] = []

    def add_trajectories(self, states: List[torch.Tensor], mus: List[torch.Tensor], times : List[Tuple[int, int]]) -> None:
        """
        Args:
            states (List[Tensor]): List of size n, tensor: [2 x k x (state dim)]
            mus (List[Tensor]): List of size n, tensor: [2],
                                relative preferences weights of states
                                (probability of picking each state, should
                                sum to 1 over the dimension with 2 values)
            times (List[Tuple[int, int]): List of size n,
                                          start times of each trajectory

        Where n is number of trajectories, k is trajectory length, and the
        state dim is whatever remaining dimensions the state has.
        """
        assert len(states) == len(mus)
        assert len(states) == len(times)

        self._states += states
        self._mus += mus
        self._times += times
        self._compute_probs()

        if self._probs[0] < self._min_sample_prob:
            del self._samples[0]
            self._compute_probs()

    def _raw_proportional_constant(self, time: int) -> float:
        return self._sample_decay**(-time)

    def _compute_probs(self):
        # could use func other than max...
        norm_const = sum(
            self._raw_proportional_constant(max(*tms))
            for tms in self._times
        )
        self._probs = [
            self._raw_proportional_constant(max(*tms)) / norm_const
            for tms in self._times
        ]

    def current_num_labels(self) -> int:
        return sum(states.size(0) for states, _ in self._samples)

    def __len__(self) -> int:
        if len(self._samples) == 0:
            return 0
        raw_size = (
            self._size_multiplier * self._samples[-1][0].size(0) /
            self._probs[-1]
        )
        if self._batch_size is not None:
            return max(round(raw_size / self._batch_size), 1) * self._batch_size
        else:
            return round(raw_size)

    # NOTE: idx input isn't relevant, a bit gross...
    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: consider instead doing this with numpy generator object and
        # accepting seed as arg
        group_idx = np.random.choice(len(self._samples), p=self._probs)
        n = self._samples[group_idx][0].size(0)
        idx = np.random.randint(0, n)

        return self._samples[group_idx][0][idx], self._samples[group_idx][1][idx]


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
