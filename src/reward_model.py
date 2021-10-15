from random import randint
from typing import Dict, List, Tuple
import gym
import torch
from torch.nn.functional import log_softmax, softmax

from utils import pfrl_trajectory_key_to_tensor


def sample_trajectory_segments_from_trajectory(
    k: int, n: int, trajectory: List[Dict[str, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        k (int): len of trajectory_seg 
        n (int): number of trajectory_seg to sample (with replacement)
        trajectory (List[Dict[str, Tensor]]): list of previous N trajectory info (of batch M),
        where 
            - N is the number of steps between updating the reward model
            - M is the number of envs
        so trajectory is list of tensors of M x <> shape 

    Returns:
        segs (Tensor): tensor of shape n x 2 x k x (M x shape of state)
        rews (Tensor): tensor of shape n x 2 x k x (M x 1)
    """
    if len(trajectory) <= k:
        raise ValueError(
            f"k = {k} is less than interval between reward_model_updates"
        )

    # N x M x shape tensor
    states = torch.cat([b['states'] for b in trajectory], dim=0)
    shp = states.shape
    states = torch.reshape(states, (shp[0] * shp[1],) + shp[2:])

    env_rewards = torch.cat([b['env_reward'] for b in trajectory], dim=0)
    shp = env_rewards.shape
    env_rewards = torch.reshape(env_rewards, (shp[0] * shp[1],) + shp[2:])

    # now env_rew and states are flattened (no extra M dimension)

    segs = []
    rews = []
    for _ in range(n):
        seg_pair = []
        rew_pair = []
        for _ in range(2):
            idx = randint(0, len(trajectory) - k)
            seg_pair.append(states[idx:])
            rew_pair.append(env_rewards[idx:])

        segs.append(torch.cat(seg_pair, dim=0))
        rews.append(torch.cat(rew_pair, dim=0))

    return torch.cat(segs, dim=0), torch.cat(rews, dim=0)


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


def compare_via_ground_truth(env_rews: torch.Tensor) -> torch.Tensor:
    """
    Args:
        env_rews (Tensor): n x 2 x k x 1

    Returns:
        (Tensor): [n] of 0 or 1, depending on which was higher reward
    """
    return torch.argmax(env_rews.sum(dim=(1, 2, 3)), dim=1)


def compare_via_human(
    trajectory_seg_1: List[Dict[str, torch.Tensor]],
    trajectory_seg_2: List[Dict[str, torch.Tensor]]
) -> float:
    raise NotImplementedError()
    return 0.0
