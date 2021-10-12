from typing import Any, List, Tuple
import random

import numpy as np
import torch
import pfrl
from pfrl.agents import PPO


class Agent(PPO):
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
                 batch_states=pfrl.utils.batch_states,
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
        self.t = 0

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
