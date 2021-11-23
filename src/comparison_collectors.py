import multiprocessing
import os
import os.path as osp
import uuid

import torch
from torch.nn.functional import softmax

from video import write_segment_to_video, upload_to_gcs


def _compare_via_ground_truth_softmax(env_rews: torch.Tensor) -> torch.Tensor:
    """
    Args:
        env_rews (Tensor): [n x 2 x k x 1]

    Returns:
        (Tensor): [n x 2] of probs

    computes probability_of_preferring_trajectory via ground truth
    """
    return softmax(env_rews.sum(dim=(2, 3)), dim=1)


def _compare_via_ground_truth_human_like(
    env_rews: torch.Tensor, human_error_rate
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


class SyntheticComparisonCollector:
    def __init__(self, human_like_approach, human_error_rate=0.1):
        self._human_like = human_like_approach
        self._human_error_rate = human_error_rate
        self._trajectories = []
        self._mus = []
        self._times = []

    def add_values(self, trajectory_segments, env_rews, times, _):
        assert len(trajectory_segments) == len(env_rews)
        assert len(trajectory_segments) == len(times)

        if self._human_like:
            mus = _compare_via_ground_truth_human_like(
                env_rews, human_error_rate=self._human_error_rate
            )
        else:
            mus = _compare_via_ground_truth_softmax(env_rews)

        self._trajectories += list(trajectory_segments)
        self._mus += list(mus)
        self._times += times

    def __len__(self):
        return len(self._trajectories)

    @property
    def labeled_comparisons(self):
        return self._trajectories, self._mus, self._times

    def n_labeled_comparisons(self):
        return len(self.labeled_comparisons[0])

    def poll_for_labels(self):
        pass

    def pop_labeled(self):
        out = self.labeled_comparisons

        self._trajectories = []
        self._mus = []
        self._times = []

        return out


def _write_and_upload_video(env, gcs_path, local_path, obs):
    write_segment_to_video(obs, fname=local_path, env=env)
    upload_to_gcs(local_path, gcs_path)


class HumanComparisonCollector:
    def __init__(self, experiment_name, env, human_error_rate):
        from human_feedback_api import Comparison

        self._comparisons = []
        self.experiment_name = experiment_name
        self.env = env
        self._upload_workers = multiprocessing.Pool(4)
        self._human_error_rate = human_error_rate

        if Comparison.objects.filter(experiment_name=experiment_name
                                    ).count() > 0:
            raise EnvironmentError(
                "Existing experiment named %s! Pick a new experiment name." %
                experiment_name
            )

    def convert_segment_to_media_url(self, comparison_uuid, side, obs):
        tmp_media_dir = '/tmp/learning_from_human_preferences_media'
        media_id = "%s-%s.mp4" % (comparison_uuid, side)
        local_path = osp.join(tmp_media_dir, media_id)
        gcs_bucket = os.environ.get('LEARNING_FROM_HUMAN_PREFS_GCS_BUCKET')
        assert gcs_bucket is not None
        gcs_path = osp.join(gcs_bucket, media_id)
        self._upload_workers.apply_async(
            _write_and_upload_video, (self.env, gcs_path, local_path, obs)
        )

        media_url = "https://storage.googleapis.com/%s/%s" % (
            gcs_bucket.lstrip("gs://"), media_id
        )
        return media_url

    def _create_comparison_in_webapp(self, left_obs, right_obs):
        """Creates a comparison DB object. Returns the db_id of the comparison"""
        from human_feedback_api import Comparison

        comparison_uuid = str(uuid.uuid4())
        comparison = Comparison(
            experiment_name=self.experiment_name,
            media_url_1=self.convert_segment_to_media_url(
                comparison_uuid, 'left', left_obs
            ),
            media_url_2=self.convert_segment_to_media_url(
                comparison_uuid, 'right', right_obs
            ),
            response_kind='left_or_right',
            priority=1.
        )
        comparison.full_clean()
        comparison.save()
        return comparison.id

    def add_values(self, trajectory_segments, env_rews, times, human_obs):
        assert len(trajectory_segments) == len(env_rews)
        assert len(trajectory_segments) == len(times)
        assert len(trajectory_segments) == len(human_obs)

        for traj, time, obs in zip(trajectory_segments, times, human_obs):
            self._comparisons.append(
                [
                    traj,
                    None,
                    time,
                    self._create_comparison_in_webapp(obs[0], obs[1])
                ]
            )

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [
            list(x) for x in
            zip(*[comp for comp in self._comparisons if comp[1] is not None])
        ]

    def n_labeled_comparisons(self):
        return len(self.labeled_comparisons[0])

    def _unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp[1] is None]

    def pop_labeled(self):
        self.poll_for_labels()
        out = self.labeled_comparisons
        self._comparisons = self._unlabeled_comparisons()
        return out

    def poll_for_labels(self):
        from human_feedback_api import Comparison

        for comparison in self._unlabeled_comparisons():
            db_comp = Comparison.objects.get(pk=comparison[3])
            if db_comp.response == 'left':
                mu = [1 - self._human_error_rate, self._human_error_rate]
            elif db_comp.response == 'right':
                mu = [self._human_error_rate, 1 - self._human_error_rate]
            elif db_comp.response == 'tie' or db_comp.response == 'abstain':
                mu = [0.5, 0.5]
            else:
                # If we did not match, then there is no response yet, so we just wait
                mu = None
            comparison[1] = torch.tensor(mu)


if __name__ == "__main__":
    print(
        _compare_via_ground_truth_human_like(
            torch.tensor(
                [
                    [100.0, 0.0],
                    [0.0, 0.0],
                    [0.01, 0.0],
                    [0.1, 0.0],
                    [0.0, 0.1],
                    [0.0, 0.1],
                ]
            ).unsqueeze(-1).unsqueeze(-1),
            human_error_rate=0.1,
        )
    )
