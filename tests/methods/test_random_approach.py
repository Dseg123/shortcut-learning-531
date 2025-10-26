"""Test base random approach."""

import pytest

from shortcut_learning.configs import (
    ApproachConfig,
    CollectionConfig,
    EvaluationConfig,
    PolicyConfig,
    TrainingConfig,
)
from shortcut_learning.methods.pipeline import Metrics, pipeline_from_configs
from shortcut_learning.problems.obstacle2d.system import BaseObstacle2DTAMPSystem


@pytest.mark.parametrize(
    "system_cls",
    [BaseObstacle2DTAMPSystem],
)
def test_random_approach(system_cls):
    """Test random approach on different environments."""
    system = system_cls.create_default(seed=42)

    approach_config = ApproachConfig(approach_type="random", approach_name="example")

    policy_config = PolicyConfig(policy_type="rl_ppo")

    collect_config = CollectionConfig()
    train_config = TrainingConfig()
    eval_config = EvaluationConfig()

    metrics = pipeline_from_configs(
        system,
        approach_config,
        policy_config,
        collect_config,
        train_config,
        eval_config,
    )

    print(metrics)

    assert isinstance(metrics, Metrics)

    # approach = RandomApproach(system, seed=42)

    # steps = run_episode(system, approach, max_steps)
    # assert steps <= max_steps


if __name__ == "__main__":
    test_random_approach(BaseObstacle2DTAMPSystem)
