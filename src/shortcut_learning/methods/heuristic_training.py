"""Framework for training a heuristic distance model.

This module implements the following:
1. Calculates shortest distance between shortcut pairs
2. Stores this data to train a heuristic distance model
3. Returns the trained heuristic distance model
"""

from __future__ import annotations

import math
from typing import TypeVar

import gymnasium as gym
import torch

from shortcut_learning.methods.heuristic_policies.goal_conditioned_rl import (
    GoalConditionedDQN,
    train_goal_conditioned_rl,
)
from shortcut_learning.methods.wrappers import SLAPWrapperV2
from shortcut_learning.methods.training_data import ShortcutTrainingData

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def heuristic_training(
    env: gym.Env,
    training_data: ShortcutTrainingData,
) -> GoalConditionedDQN:
    """Given ShortcutTrainingData, run GoalConditioned DQN 
    over each shortcut pair to determine its path length. We will use this path
    length to train our agent.

    Args:
        env: The gymnasium environment
        heuristic_model: Model that returns heuristic distance
                         between two PlanningGraphNodes
        shortcut_pairs: list of shortcut pairs

    Returns:
        Trained DQN agent
    """

    perceiver = env._extract_perceiver(env)

    slap_env_wrapper = SLAPWrapperV2(env, perceiver,  
                                     max_episode_steps=500,
                                     step_penalty=-1,
                                     achievement_bonus=0,
                                     )
    slap_env_wrapper.configure_training(training_data)
    
    # Initialize agent
    state_dim = slap_env_wrapper.observation_space.shape[0]
    action_dim = slap_env_wrapper.action_space.n

    agent = GoalConditionedDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Train
    print("Training goal-conditioned RL agent...")
    _, _ = train_goal_conditioned_rl(slap_env_wrapper, agent, num_episodes=500)

    return agent
