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

from shortcut_learning.methods.graph_utils import PlanningGraphNode
from shortcut_learning.methods.heuristic_models.base_model import HeuristicModel
from shortcut_learning.methods.heuristic_policies.goal_conditioned_rl import (
    GoalConditionedDQN,
    GoalConditionedWrapper,
    train_goal_conditioned_rl,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def heuristic_training(
    env: gym.Env,
    heuristic_model: HeuristicModel,
    shortcut_pairs: list[tuple[PlanningGraphNode, PlanningGraphNode]],
) -> HeuristicModel:
    """Given ShortcutTrainingData and a heuristic model, run an RL algorithm
    over each shortcut pair to determine its path length. We will use this path
    length to train our heuristic model.

    Args:
        env: The gymnasium environment
        heuristic_model: Model that returns heuristic distance
                         between two PlanningGraphNodes
        shortcut_pairs: list of shortcut pairs

    Returns:
        Trained heuristic model
    """

    training_data = []

    for shortcut_pair in shortcut_pairs:
        heuristic_distance = _calculate_path_length_heuristic(env, shortcut_pair)
        training_data.append(heuristic_distance)

    heuristic_model.train(training_data)

    return heuristic_model


def _calculate_path_length_heuristic(
    env: gym.Env, shortcut_pair: tuple[PlanningGraphNode, PlanningGraphNode]
) -> float:
    """Given an PlanningGraphNode pair, run rollouts with RL across
    corresponding low-level states to determine the plan length heuristic.

    We will enumerate over all low level states corresponding to the
    first abstract state and run GCRL to calculate the value function (distance function)
    to get to the second abstract state. This value will be used to determine our heuristic.
    We will take the most optimal (minimum) path distance start from
    each of the initial low level states corresponding to the first abstract state.

    Args:
        env: The gymnasium environment
        shortcut_pair: Tuple of initial abstract state and goal abstract state

    Returns:
        Path Length between shortcut pair (float)
    """

    start_node, end_node = shortcut_pair[0], shortcut_pair[1]

    env = GoalConditionedWrapper(env, planning_graph_goal=end_node)

    # For each state, run rollouts with RL to get to the end node
    optimistic_path_distance = math.inf
    for low_level_state in start_node.states:

        # Initialize agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = GoalConditionedDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=1e-3,
            gamma=0.99,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Train
        print("Training goal-conditioned RL agent...")
        _, _ = train_goal_conditioned_rl(env, agent, num_episodes=500)

        # After training, query minimum path distance
        shortest_path_distance = agent.get_min_path_distance(low_level_state, end_node)
        optimistic_path_distance = min(optimistic_path_distance, shortest_path_distance)

    return optimistic_path_distance
