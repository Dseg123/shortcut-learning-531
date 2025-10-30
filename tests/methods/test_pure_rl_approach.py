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


def run_episode(system, approach, max_steps):
    """Run single episode with approach."""
    obs, info = system.reset()
    step_result = approach.reset(obs, info)

    # Process first step
    obs, reward, terminated, truncated, info = system.env.step(step_result.action)
    if terminated or truncated:
        return 1

    # Process remaining steps
    for step in range(1, max_steps):
        step_result = approach.step(obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = system.env.step(step_result.action)
        if terminated or truncated:
            return step + 1
    return max_steps


@pytest.mark.parametrize(
    "system_cls",
    [BaseObstacle2DTAMPSystem],
)
def test_pure_rl_approach(system_cls):
    """Test random approach on different environments."""
    system = system_cls.create_default(seed=42)

    approach_config = ApproachConfig(approach_type="pure_rl", approach_name="example")

    policy_config = PolicyConfig(policy_type="rl_ppo")

    collect_config = CollectionConfig()
    train_config = TrainingConfig(runs_per_shortcut=10)
    eval_config = EvaluationConfig(num_episodes=1)

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
    test_pure_rl_approach(BaseObstacle2DTAMPSystem)
