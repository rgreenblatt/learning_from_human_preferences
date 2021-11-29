from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RewardModelTrainingDataset(Dataset):
    """
    Dataset which allows for sliding window of states and preferences.

    We sample older trajectories exponentially less frequently. Precisely, a
    group of trajectories with time i ago is sampled with probability
    proportional to sample_decay**i.

    To save on memory, if the probability of sampling from a group of
    trajectories is less than min_sample_prob, that group is eliminated.

    The size is arbitrary because we always sample randomly, but
    it is computed as ??? (TODO, maybe change this)
    Dividing and multiplying by batch_size ensures the size is a multiple
    of batch size. This is only done if batch size isn't none.

    Args:
        sample_decay (float): see above
        min_sample_prob (float): see above
        epoch_size_multiplier (float): ??? (use this?)
    """
    def __init__(
        self,
        sample_decay: float = 0.99999,
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

        self._samples: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        self._probs: List[float] = []

    def _times_to_weight(self, l: int, r: int) -> float:
        # could use func other than max...
        return self._sample_decay**(-max(l, r))

    def add_trajectories(
        self,
        states: List[torch.Tensor],
        mus: List[torch.Tensor],
        times: List[Tuple[int, int]]
    ) -> None:
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

        # not exactly efficient...
        self._samples += [
            (state, mu, self._times_to_weight(*tms)) for state,
            mu,
            tms in zip(states, mus, times)
        ]
        self._samples.sort(key=lambda x: x[2])
        self._compute_probs()

        remaining = 1.
        i = 0

        for i, prob in enumerate(self._probs):
            if prob / remaining >= self._min_sample_prob:
                break
            remaining -= prob
            assert remaining > 0.

        del self._samples[:i]
        self._compute_probs()

    def _compute_probs(self):
        norm_const = sum(weight for _, _, weight in self._samples)
        self._probs = [weight / norm_const for _, _, weight in self._samples]

    def current_num_labels(self) -> int:
        return len(self._samples)

    def __len__(self) -> int:
        if len(self._samples) == 0:
            return 0
        raw_size = len(self._samples)  # TODO: this could be more efficient!
        if self._batch_size is not None:
            return max(round(raw_size / self._batch_size), 1) * self._batch_size
        else:
            return round(raw_size)

    # NOTE: idx input isn't relevant, a bit gross...
    def __getitem__(self, _) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: consider instead doing this with numpy generator object and
        # accepting seed as arg
        idx = np.random.choice(len(self._samples), p=self._probs)

        return self._samples[idx][0], self._samples[idx][1]


if __name__ == "__main__":
    dataset = RewardModelTrainingDataset(min_sample_prob=5e-3)

    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0
    )

    print("DataLoader tests")
    for batched in dataloader:
        print(batched.shape)
        print(batched[0].size(), batched[1].size())

    dataset.add_trajectories(
        [torch.rand(2, 1, 1) for _ in range(10)],
        [torch.full((2,), 0.5) for _ in range(10)],
        [(i * 10000, i * 10000) for i in range(10)],
    )
    print(dataset._probs)
    print(len(dataset))
    dataset.add_trajectories(
        [torch.rand(2, 1, 1) for _ in range(10, 100)],
        [torch.full((2,), 0.5) for _ in range(10, 100)],
        [(i * 10000, i * 10000) for i in range(10, 100)],
    )
    print(dataset._probs)
    print(len(dataset))

    # sampling??
    for _ in range(100):
        dataset[0]

    print("sampling after adding tests")
    for batched in dataloader:
        print(batched[0].size(), batched[1].size())
