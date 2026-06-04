"""
CartPole Scenario - A VMAS-compatible wrapper for the CartPole environment.

This module provides a scenario that integrates Gymnasium's CartPole environment
with the VMAS framework using continuous action outputs with argmax discretization.

Key Features:
    - Uses CartPole-v1 from Gymnasium
    - Continuous action space: 2 continuous logits, argmax selected to [0, 1]
    - 4-dimensional observations: [position, velocity, angle, angular_velocity]
    - Single agent scenario
    - Episodes: 500 steps (truncated)

"""

import gymnasium as gym
import torch
import numpy as np
from vmas.simulator.core import Agent
from vmas.simulator.utils import Color
from .BaseGymScenario import BaseGymScenario


class Scenario(BaseGymScenario):
    """
    VMAS Scenario wrapper for the CartPole environment.
    
    The CartPole environment simulates a pole balanced on a moving cart.
    The goal is to keep the pole upright by applying left/right forces to the cart.
    
    Action Mode:
        Uses 2 continuous action outputs with argmax selection:
        - action[0]: PUSH_LEFT (Atari action 0)
        - action[1]: PUSH_RIGHT (Atari action 1)
        The discrete action is selected as argmax(action_values).
    
    Attributes:
        observation_size: 4 ([position, velocity, angle, angular_velocity])
        action_size: 2 (two continuous logits, argmax selected)
        agents: 1 (the cart pole controller)
    """
    
    
    def _get_observation_dim(self) -> int:
        """Return observation dimension for CartPole: [position, velocity, angle, angular_velocity]"""
        return 4
    
    def _create_gym_env(self, render_mode):
        """Create a single CartPole-v1 environment."""
        return gym.make("CartPole-v1", render_mode=render_mode)
    
    def _get_agent_config(self) -> tuple:
        """Return (name, color) for CartPole agent."""
        return ("cartpole", Color.RED)
    
    def _initialize_buffers(self, num_envs: int):
        """Initialize action buffer for CartPole."""
        # CartPole expects (num_envs,) shaped discrete action array
        self._gym_actions_buffer = np.zeros(num_envs, dtype=np.int32)
    
    def _process_agent_action(self, agent: Agent):
        """Convert 2 continuous actions to discrete via argmax."""
        # agent.action.u is (num_envs, 2), two continuous actions in [-1, 1]
        # Keep on GPU for argmax operation
        indices = torch.argmax(agent.action.u, dim=1)
        
        # Convert indices to numpy and store in buffer
        self._gym_actions_buffer[:] = indices.cpu().numpy().astype(np.int32)
    
    def _get_gym_actions(self):
        """Return action buffer for gym.step()."""
        return self._gym_actions_buffer

    def max_rewards(self):
        """Maximum possible reward (500 steps)."""
        return torch.full((self.num_envs,), 500.0, device=self.device)
