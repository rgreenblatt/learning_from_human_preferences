from random import randint
from typing import Callable, Dict, List, Tuple
import torch
from torch.nn.functional import log_softmax, softmax


def sample_trajectory_segments_from_trajectory(
    k: int,
    n: int,
    trajectory: List[Tuple[Dict[str, torch.Tensor], int]],
    state_batch_func: Callable,
    rew_batch_func: Callable,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Args:
        k (int): len of trajectory_seg
        n (int): number of trajectory_seg to sample (with replacement)
        trajectory (List[Dict[str, Tensor]]): list of previous N trajectory info,
        where
            - N is the number of steps between updating the reward model
        so trajectory is list of tensors of M x <> shape

    Returns:
        segs (Tensor): tensor of shape n x 2 x k x (shape of state)
        rews (Tensor): tensor of shape n x 2 x k x (1)
        start times List[Tuple[int, int]]: length n
    """
    if len(trajectory) <= k:
        raise ValueError(
            f"k = {k} is less than number of things in trajectory {len(trajectory)}"
        )

    # N x shape tensor
    states = state_batch_func([b[0]['state'] for b in trajectory])
    env_rewards = rew_batch_func([b[0]['env_reward'] for b in trajectory])

    segs = []
    rews = []
    times = []
    for _ in range(n):
        seg_pair = []
        rew_pair = []
        time_pair = []
        for _ in range(2):
            idx = randint(0, (len(trajectory) // k) - 1) * k
            seg_pair.append(states[idx:idx + k])
            rew_pair.append(env_rewards[idx:idx + k])
            time_pair.append(trajectory[idx][1])

        segs.append(torch.stack(seg_pair, dim=0))
        rews.append(torch.stack(rew_pair, dim=0))
        times.append(tuple(time_pair))

    return torch.stack(segs, dim=0), torch.stack(rews, dim=0).unsqueeze(-1), times


def probability_of_preferring_trajectory(
    reward_sum_1: torch.Tensor, reward_sum_2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        reward_sum_1 (Tensor): [n]
        reward_sum_2 (Tensor): [n]

    Returns:
        log_probs (Tensor): [n x 2]
        probs (Tensor): [n x 2]

    Implements the calculation of the probability for preferring [1 > 2, 2 > 1]
    from r_hat rewards

    """
    rews = torch.cat((reward_sum_1, reward_sum_2), dim=1)
    probs = softmax(rews, dim=1)
    log_probs = log_softmax(rews, dim=1)
    return log_probs, probs
