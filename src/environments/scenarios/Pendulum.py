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
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import torch
import numpy as np
from vmas.simulator.core import Agent, World, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
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
    
    def make_world(self, num_envs: int, device: torch.device, **kwargs) -> World:
        """
        Create and initialize the VMAS world with vectorized Pendulum environments.
        
        Args:
            num_envs: Number of parallel environments
            device: Device for tensor computations
            **kwargs:
                - render_mode: Rendering mode (None, "human")
                - async_vectorize: Use async environments (default: num_envs >= 16)
        """
        self.num_envs = num_envs
        self.device = device
        self.observation_dim = 3  # [cos(theta), sin(theta), theta_dot]
        
        # Handle render mode
        render_mode_raw = kwargs.get("render_mode", None)
        if render_mode_raw is True:
            self.render_mode = "human"
        elif render_mode_raw is False:
            self.render_mode = None
        else:
            self.render_mode = render_mode_raw

        def make_env(render_mode=None):
            """Create a single Pendulum environment."""
            return gym.make(
                "Pendulum-v1",
                render_mode=render_mode
            )
        
        def make_env_with_render():
            return make_env(self.render_mode)

        # Create vectorized environments
        if self.render_mode == "human":
            env_fns = [make_env_with_render] + [make_env for _ in range(num_envs - 1)]
        else:
            env_fns = [make_env for _ in range(num_envs)]
        
        use_async = kwargs.get("async_vectorize", num_envs >= 16)
        if use_async:
            self.gym_env = AsyncVectorEnv(env_fns)
        else:
            self.gym_env = SyncVectorEnv(env_fns)

        # Initialize state tensors
        self.current_rewards = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # Observation cache: (num_envs, 3)
        self.obs_cache = torch.zeros(
            num_envs, self.observation_dim,
            device=device,
            dtype=torch.float32
        )
        
        # Pre-allocated buffer for actions with shape (num_envs, 1)
        # Pendulum expects 1D array per action, not scalar
        self._actions_buffer = np.zeros((num_envs, 1), dtype=np.float32)

        # Create VMAS world (minimal placeholder - physics disabled)
        world = World(
            batch_dim=num_envs,
            device=device,
            x_semidim=1,
            y_semidim=1,
            collision_force=0,
            substeps=1,
            drag=0,
            linear_friction=0,
            angular_friction=0,
        )

        # Agent is non-movable/non-rotatable to skip physics in World.step()
        # Extract action_size from kwargs (passed via vmas.make_env)
        action_size_param = kwargs.get("action_size", 1)  # Default to 1 for Pendulum
        agent = Agent(
            name="pendulum",
            shape=Sphere(radius=0.05),
            color=Color.GREEN,
            movable=False,      # Skip force/velocity integration
            rotatable=False,    # Skip torque/angular integration
            collide=False,      # Skip collision detection
            action_size=action_size_param,  # 1 continuous action (torque)
        )
        world.add_agent(agent)
        
        print(f"✓ Pendulum initialized: {num_envs} envs, {self.observation_dim}-dim observations")
        
        return world
    
    def reset_world_at(self, env_index: int = None):
        """Reset environment(s) to initial state."""
        obs, _ = self.gym_env.reset()
        self._update_obs_cache(obs)
        self.current_rewards.zero_()
        self.dones.zero_()
    
    def _update_obs_cache(self, obs_array):
        """
        Update observation cache with Pendulum state.
        
        Observations are already in [-1, 1] range for cos/sin, theta_dot in [-8, 8].
        """
        self.obs_cache.copy_(
            torch.from_numpy(obs_array.astype(np.float32)).to(self.device)
        )
    
    def reward(self, agent: Agent):
        """Return current rewards."""
        return self.current_rewards
    
    def observation(self, agent: Agent):
        """
        Return Pendulum observations.
        
        Returns:
            Tensor of shape (num_envs, 3) with [cos(theta), sin(theta), theta_dot]
        """
        return self.obs_cache
    
    def done(self):
        """Return done flags."""
        return self.dones
    
    def process_action(self, agent: Agent):
        """
        Apply continuous torque actions to the Pendulum environments.
        
        Actions are continuous values in [-1, 1], scaled to torque [-2, 2].
        """
        # agent.action.u is (num_envs, 1), continuous [-1, 1]
        action_values = agent.action.u[:, 0].cpu().numpy()
        
        # Scale to [-2, 2] and store in pre-allocated buffer
        # Shape (num_envs, 1) because Pendulum expects 1D array per action
        self._actions_buffer[:, 0] = action_values * 2.0
        
        # Step environments with pre-allocated buffer
        obs, rewards, terminateds, truncateds, _ = self.gym_env.step(self._actions_buffer)

        # Update caches
        self._update_obs_cache(obs)
        self.current_rewards.copy_(
            torch.from_numpy(rewards.astype(np.float32)).to(self.device)
        )
        
        dones = terminateds | truncateds
        self.dones.copy_(
            torch.from_numpy(dones).to(self.device)
        )

    def max_rewards(self):
        """Maximum possible reward (perfect balance at upright position)."""
        # Pendulum reward = -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
        # Maximum reward is 0 (when theta=0, theta_dot=0, torque=0)
        return torch.zeros((self.num_envs,), device=self.device)
    
    def diffreward(self, prevs, acts, nexts):
        """Not implemented - Pendulum rewards are not differentiable in this wrapper."""
        raise NotImplementedError("Pendulum has non-differentiable rewards in this wrapper")
    
    def zero_grad(self):
        """Clear gradients (no-op for non-differentiable env)."""
        pass