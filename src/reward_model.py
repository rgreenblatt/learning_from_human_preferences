from random import randint
from typing import Callable, Dict, List, Tuple
import torch
from torch.nn.functional import log_softmax, softmax


def sample_trajectory_segments_from_trajectory(
    k: int,
    n: int,
    trajectory: List[Dict[str, torch.Tensor]],
    state_batch_func: Callable,
    rew_batch_func: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """
    if len(trajectory) <= k:
        raise ValueError(
            f"k = {k} is less than number of things in trajectory {len(trajectory)}"
        )

    # N x shape tensor
    states = state_batch_func([b['state'] for b in trajectory])
    env_rewards = rew_batch_func([b['env_reward'] for b in trajectory])

    segs = []
    rews = []
    for _ in range(n):
        seg_pair = []
        rew_pair = []
        for _ in range(2):
            idx = randint(0, (len(trajectory) // k) - 1) * k
            seg_pair.append(states[idx:idx + k])
            rew_pair.append(env_rewards[idx:idx + k])

        segs.append(torch.stack(seg_pair, dim=0))
        rews.append(torch.stack(rew_pair, dim=0))

    return torch.stack(segs, dim=0), torch.stack(rews, dim=0).unsqueeze(-1)


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


def compare_via_ground_truth_softmax(env_rews: torch.Tensor) -> torch.Tensor:
    """
    Args:
        env_rews (Tensor): [n x 2 x k x 1]

    Returns:
        (Tensor): [n x 2] of probs

    computes probability_of_preferring_trajectory via ground truth
    """
    return softmax(env_rews.sum(dim=(2, 3)), dim=1)


def compare_via_ground_truth_human_like(
    env_rews: torch.Tensor, human_error_rate=0.1
) -> torch.Tensor:
    """
    Args:
        env_rews (Tensor): [n x 2 x k x 1]

    Returns:
        (Tensor): [n x 2] of probs

    computes probability_of_preferring_trajectory via ground truth
    """
    rew_sums = env_rews.sum(dim=(2, 3))
    rew_abs_diffs = torch.abs(rew_sums[:, 0] - rew_sums[:, 1])

    lower_prob = torch.full(
        (env_rews.size(0),),
        human_error_rate,
        dtype=env_rews.dtype,
        device=env_rews.device
    )
    higher_prob = torch.full_like(lower_prob, 1 - human_error_rate)
    left_better = torch.stack([higher_prob, lower_prob], dim=1)
    right_better = torch.stack([lower_prob, higher_prob], dim=1)
    distinct_probs = torch.where(
        (rew_sums[:, 0] > rew_sums[:, 1]).unsqueeze(-1),
        left_better,
        right_better
    )

    epsilon = 0.05
    # any reason to change epsilon? (it's just supposed to indicate when
    # a trajectory is 'clearly' better.)
    return torch.where(
        (rew_abs_diffs > epsilon).unsqueeze(-1),
        distinct_probs,
        torch.full_like(distinct_probs, 0.5)
    )


def compare_via_human(
    trajectory_seg_1: List[Dict[str, torch.Tensor]],
    trajectory_seg_2: List[Dict[str, torch.Tensor]]
) -> float:
    raise NotImplementedError()
    return 0.0


if __name__ == "__main__":
    print(
        compare_via_ground_truth_human_like(
            torch.tensor(
                [
                    [100.0, 0.0],
                    [0.0, 0.0],
                    [0.01, 0.0],
                    [0.1, 0.0],
                    [0.0, 0.1],
                    [0.0, 0.1],
                ]
            ).unsqueeze(-1).unsqueeze(-1)
        )
    )
