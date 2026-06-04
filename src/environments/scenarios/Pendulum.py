"""
Pendulum Scenario - A VMAS-compatible wrapper for the Pendulum environment.

This module provides a scenario that integrates the Gymnasium Pendulum
environment with the VMAS framework. The Pendulum task involves swinging
a pendulum up and balancing it at the top using continuous torque control.

Key Features:
    - Uses Pendulum-v1 from Gymnasium
    - Continuous action space: agent actions in [-1, 1] mapped to torque [-2, 2]
    - 3-dimensional observations: [cos(theta), sin(theta), theta_dot]
    - Single agent scenario
    - Episodes: 200 steps (truncated)

"""

import gymnasium as gym
import torch
import numpy as np
from vmas.simulator.core import Agent
from vmas.simulator.utils import Color
from .BaseGymScenario import BaseGymScenario


class Scenario(BaseGymScenario):
    """
    VMAS Scenario wrapper for the Pendulum environment.
    
    The Pendulum environment simulates a pendulum attached to a fixed point
    that can be controlled by applying torque. The goal is to swing the pendulum
    up to the upright position (theta = 0) and balance it there.
    
    Attributes:
        observation_size: 3 ([cos(theta), sin(theta), theta_dot])
        action_size: 1 (continuous action in [-1, 1], mapped to torque [-2, 2])
        agents: 1 (the pendulum controller)
    """
    
    
    def _get_observation_dim(self) -> int:
        """Return observation dimension for Pendulum: [cos(theta), sin(theta), theta_dot]"""
        return 3
    
    def _create_gym_env(self, render_mode):
        """Create a single Pendulum-v1 environment."""
        return gym.make("Pendulum-v1", render_mode=render_mode)
    
    def _get_agent_config(self) -> tuple:
        """Return (name, color) for Pendulum agent."""
        return ("pendulum", Color.GREEN)
    
    def _initialize_buffers(self, num_envs: int):
        """Initialize action buffer for Pendulum."""
        # Pendulum expects (num_envs, 1) shaped action array
        self._actions_buffer = np.zeros((num_envs, 1), dtype=np.float32)
    
    def _process_agent_action(self, agent: Agent):
        """Convert continuous agent action to Pendulum torque format."""
        # agent.action.u is (num_envs, 1), continuous [-1, 1]
        action_values = agent.action.u[:, 0].cpu().numpy()
        # Scale to [-2, 2] and store in buffer
        self._actions_buffer[:, 0] = action_values * 2.0
    
    def _get_gym_actions(self):
        """Return action buffer for gym.step()."""
        return self._actions_buffer

    def max_rewards(self):
        """Maximum possible reward (perfect balance at upright position)."""
        # Pendulum reward = -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
        # Maximum reward is 0 (when theta=0, theta_dot=0, torque=0)
        return torch.zeros((self.num_envs,), device=self.device)