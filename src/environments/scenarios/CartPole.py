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
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import torch
import numpy as np
from vmas.simulator.core import Agent, World, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
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
    
    def make_world(self, num_envs: int, device: torch.device, **kwargs) -> World:
        """
        Create and initialize the VMAS world with vectorized CartPole environments.
        
        Args:
            num_envs: Number of parallel environments
            device: Device for tensor computations
            **kwargs:
                - render_mode: Rendering mode (None, "human")
                - async_vectorize: Use async environments (default: num_envs >= 16)
        """
        self.num_envs = num_envs
        self.device = device
        self.observation_dim = 4  # [position, velocity, angle, angular_velocity]
        
        # Handle render mode
        render_mode_raw = kwargs.get("render_mode", None)
        if render_mode_raw is True:
            self.render_mode = "human"
        elif render_mode_raw is False:
            self.render_mode = None
        else:
            self.render_mode = render_mode_raw

        def make_env(render_mode=None):
            """Create a single CartPole environment."""
            return gym.make(
                "CartPole-v1",
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
        
        # Observation cache: (num_envs, 4)
        # Observations from CartPole are already normalized reasonably
        self.obs_cache = torch.zeros(
            num_envs, self.observation_dim,
            device=device,
            dtype=torch.float32
        )
        
        # Pre-allocated buffer for actions with shape (num_envs,)
        # CartPole expects discrete actions [0, 1]
        self._gym_actions_buffer = np.zeros(num_envs, dtype=np.int32)

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
        action_size_param = kwargs.get("action_size", 1)  # Default to 1 (single continuous action)
        agent = Agent(
            name="cartpole",
            shape=Sphere(radius=0.05),
            color=Color.RED,
            movable=False,      # Skip force/velocity integration
            rotatable=False,    # Skip torque/angular integration
            collide=False,      # Skip collision detection
            action_size=action_size_param,  # CRITICAL: tells VMAS we have N continuous actions
        )
        world.add_agent(agent)
        
        print(f"✓ CartPole initialized: {num_envs} envs, {self.observation_dim}-dim observations")
        
        return world
    
    def reset_world_at(self, env_index: int = None):
        """Reset environment(s) to initial state."""
        obs, _ = self.gym_env.reset()
        self._update_obs_cache(obs)
        self.current_rewards.zero_()
        self.dones.zero_()
    
    def _update_obs_cache(self, obs_array):
        """
        Update observation cache with CartPole state.
        
        Observations are: [position, velocity, angle, angular_velocity]
        Already in reasonable ranges, no normalization needed.
        """
        self.obs_cache.copy_(
            torch.from_numpy(obs_array.astype(np.float32)).to(self.device)
        )
    
    def reward(self, agent: Agent):
        """Return current rewards."""
        return self.current_rewards
    
    def observation(self, agent: Agent):
        """
        Return CartPole observations.
        
        Returns:
            Tensor of shape (num_envs, 4) with state
        """
        return self.obs_cache
    
    def done(self):
        """Return done flags."""
        return self.dones
    
    def process_action(self, agent: Agent):
        """
        Convert 2 continuous actions to discrete CartPole actions via argmax selection.
        
        Action mapping:
            - agent.action.u[:, 0]: PUSH_LEFT (CartPole action 0)
            - agent.action.u[:, 1]: PUSH_RIGHT (CartPole action 1)
        
        Selection: argmax(action_values) determines which discrete action is executed.
        This allows the policy to output multiple action logits and let the argmax select.
        Optimized: argmax is computed on GPU, only final actions converted to numpy.
        """
        # agent.action.u is (num_envs, 2), two continuous actions in [-1, 1]
        # Keep on GPU for argmax operation
        indices = torch.argmax(agent.action.u, dim=1)  # Shape: (num_envs,), values in [0, 1]
        
        # Convert indices to numpy only for gymnasium compatibility
        indices_np = indices.cpu().numpy().astype(np.int32)
        
        # Indices are already 0 or 1, which are valid CartPole actions
        self._gym_actions_buffer[:] = indices_np

        # Step environments
        obs, rewards, terminateds, truncateds, infos = self.gym_env.step(self._gym_actions_buffer)

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
        """Maximum possible reward (500 steps)."""
        return torch.full((self.num_envs,), 500.0, device=self.device)
    
    def diffreward(self, prevs, acts, nexts):
        """Not implemented - CartPole rewards are not differentiable."""
        raise NotImplementedError("CartPole has non-differentiable rewards")
    
    def zero_grad(self):
        """Clear gradients (no-op for non-differentiable env)."""
        pass
