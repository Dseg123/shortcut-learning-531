"""Test heuristic training functionality."""

import pytest
import torch
import numpy as np

from shortcut_learning.methods.heuristic_training import heuristic_training
from shortcut_learning.methods.training_data import ShortcutTrainingData
from shortcut_learning.methods.heuristic_policies.goal_conditioned_rl import GoalConditionedDQN
from shortcut_learning.problems.obstacle2d.system import BaseObstacle2DTAMPSystem


@pytest.fixture
def mock_training_data():
    """Create mock training data for testing."""
    # Create simple training data with a few shortcut pairs
    return ShortcutTrainingData(
        shortcut_pairs=[],  # Add mock shortcut pairs here
        shortcut_distances={},  # Add mock distances here
    )


@pytest.mark.parametrize(
    "system_cls",
    [BaseObstacle2DTAMPSystem],
)
def test_heuristic_training_initialization(system_cls, mock_training_data):
    """Test that heuristic training properly initializes and returns a trained agent."""
    # Create system and environment
    system = system_cls.create_default(seed=42)
    env = system.env

    # Run heuristic training
    trained_agent = heuristic_training(env, mock_training_data)

    # Basic checks
    assert isinstance(trained_agent, GoalConditionedDQN)
    assert trained_agent.state_dim == env.observation_space.shape[0]
    assert trained_agent.action_dim == env.action_space.n

    # Check that the model has learned parameters
    for param in trained_agent.q_network.parameters():
        assert torch.any(param != 0)  # Ensure weights have been updated


@pytest.mark.parametrize(
    "system_cls",
    [BaseObstacle2DTAMPSystem],
)
def test_heuristic_training_inference(system_cls, mock_training_data):
    """Test that trained agent can make predictions."""
    # Create system and environment
    system = system_cls.create_default(seed=42)
    env = system.env

    # Train agent
    trained_agent = heuristic_training(env, mock_training_data)

    # Test inference
    obs = env.observation_space.sample()
    goal = env.observation_space.sample()
    
    # Convert to torch tensors
    state = torch.FloatTensor(np.concatenate([obs, goal])).unsqueeze(0)
    
    # Get action from agent
    with torch.no_grad():
        action = trained_agent.select_action(state)
    
    # Check action is valid
    assert env.action_space.contains(action)


if __name__ == "__main__":
    pytest.main([__file__])
