"""
Abstract base class for VMAS scenarios wrapping Gymnasium environments.

This module provides common functionality for single-agent RL environments
integrated with VMAS, handling cache management, observation handling,
and non-differentiable reward computation.

Subclasses must implement:
- make_world(): Create and initialize the VMAS world
- process_action(): Convert agent actions to environment steps
"""

from abc import abstractmethod
import torch
import numpy as np
from vmas.simulator.core import Agent, World, Sphere
from vmas.simulator.utils import Color
from vmas.simulator.scenario import BaseScenario
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv


class BaseGymScenario(BaseScenario):
    """
    Abstract base class for Gymnasium-based VMAS scenarios.
    
    Provides common functionality for:
    - Render mode handling
    - Observation caching
    - Reward and done flag management
    - Non-differentiable environment integration
    
    Subclasses must implement:
    - make_world(num_envs, device, **kwargs): Create VMAS world (template method)
    - process_action(agent): Step environment with agent actions
    - max_rewards(): Get the maximum reward obtainable from the environment
    - _get_observation_dim(): Return observation dimension (abstract)
    - _create_gym_env(render_mode): Create gymnasium environment (abstract)
    - _get_agent_config(): Return agent name, color (abstract)
    - _initialize_buffers(num_envs): Initialize environment-specific buffers (abstract)
    """
    
    def make_world(self, num_envs: int, device: torch.device, **kwargs) -> World:
        """
        Template method for creating VMAS world with vectorized gymnasium environments.
        
        Common initialization flow:
        1. Store num_envs and device
        2. Get observation dimension from subclass
        3. Handle render mode
        4. Create vectorized environments
        5. Initialize state tensors (rewards, dones, obs_cache)
        6. Initialize environment-specific buffers
        7. Create VMAS world and agent
        
        Subclasses implement abstract methods for environment-specific details.
        
        Args:
            num_envs: Number of parallel environments
            device: Device for tensor computations
            **kwargs: Additional configuration parameters
            
        Returns:
            VMAS World instance
        """
        self.num_envs = num_envs
        self.device = device
        self.observation_dim = self._get_observation_dim()
        
        # Handle render mode
        render_mode_raw = kwargs.get("render_mode", None)
        self.render_mode = self._handle_render_mode(render_mode_raw)

        # Create vectorized environments
        env_fns = self._create_env_fns(num_envs)
        use_async = kwargs.get("async_vectorize", num_envs >= 16)
        if use_async:
            self.gym_env = AsyncVectorEnv(env_fns)
        else:
            self.gym_env = SyncVectorEnv(env_fns)

        # Initialize state tensors
        self.current_rewards = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # Observation cache
        self.obs_cache = torch.zeros(
            num_envs, self.observation_dim,
            device=device,
            dtype=torch.float32
        )
        
        # Environment-specific buffer initialization
        self._initialize_buffers(num_envs)

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

        # Create agent with environment-specific configuration
        agent_name, agent_color = self._get_agent_config()
        action_size = kwargs.get("action_size", 1)  # Default to 1 if not specified
        agent = Agent(
            name=agent_name,
            shape=Sphere(radius=0.05),
            color=agent_color,
            movable=False,
            rotatable=False,
            collide=False,
            action_size=action_size,
        )
        world.add_agent(agent)
        
        print(f"✓ {agent_name.capitalize()} initialized: {num_envs} envs, {self.observation_dim}-dim observations")
        
        return world
    
    def _create_env_fns(self, num_envs: int):
        """
        Create environment functions for vectorization.
        
        Handles rendering mode (first env with rendering, rest without).
        
        Args:
            num_envs: Number of environments
            
        Returns:
            List of environment factory functions
        """
        if self.render_mode == "human":
            env_fns = [lambda: self._create_gym_env(self.render_mode)]
            env_fns += [lambda: self._create_gym_env(None) for _ in range(num_envs - 1)]
        else:
            env_fns = [lambda: self._create_gym_env(None) for _ in range(num_envs)]
        
        return env_fns
    
    @abstractmethod
    def _get_observation_dim(self) -> int:
        """
        Return the observation dimension for this environment.
        
        Returns:
            Integer observation dimension
        """
        raise NotImplementedError
    
    @abstractmethod
    def _create_gym_env(self, render_mode):
        """
        Create a single gymnasium environment instance.
        
        Subclass responsibility: create and optionally wrap the environment.
        
        Args:
            render_mode: Rendering mode (None or "human")
            
        Returns:
            Gymnasium environment instance
        """
        raise NotImplementedError
    
    @abstractmethod
    def _get_agent_config(self) -> tuple:
        """
        Return agent configuration for VMAS (name and color only).
        
        Action size is passed via kwargs to make_world, not here.
        
        Returns:
            Tuple of (agent_name: str, agent_color: Color)
        """
        raise NotImplementedError
    
    @abstractmethod
    def _initialize_buffers(self, num_envs: int):
        """
        Initialize environment-specific pre-allocated buffers.
        
        Examples: action buffers, reward modification buffers, etc.
        
        Args:
            num_envs: Number of environments
        """
        raise NotImplementedError
    
    def reset_world_at(self, env_index: int = None):
        """
        Reset environment(s) to initial state.
        
        Common reset logic for all gymnasium-based scenarios.
        Resets environments, clears caches, and zeros rewards/done flags.
        
        Args:
            env_index: Optional index (unused, for BaseScenario compatibility)
        """
        obs, _ = self.gym_env.reset()
        self._update_obs_cache(obs)
        self.current_rewards.zero_()
        self.dones.zero_()
    
    def process_action(self, agent: Agent):
        """
        Template method for processing agent actions and stepping the environment.
        
        Common flow:
        1. Convert agent actions to gymnasium-compatible format
        2. Step environment
        3. Apply environment-specific reward modifications
        4. Finalize step (update caches)
        
        Subclasses implement abstract methods for action conversion and reward mods.
        
        Args:
            agent: VMAS Agent object with action.u containing continuous outputs
        """
        # Convert agent actions to gymnasium format
        self._process_agent_action(agent)
        
        # Step environments
        obs, rewards, terminateds, truncateds, infos = self.gym_env.step(self._get_gym_actions())
        
        # Apply environment-specific reward modifications
        rewards = self._apply_reward_modifications(obs, rewards, infos)
        
        # Update caches
        self._finalize_step(obs, rewards, terminateds, truncateds)
    
    @abstractmethod
    def _process_agent_action(self, agent: Agent):
        """
        Convert VMAS agent actions to gymnasium-compatible actions.
        
        Subclass responsibility: extract agent.action.u and convert to gym format.
        Store result in appropriate buffer for gym.step().
        
        Args:
            agent: VMAS Agent with action.u (continuous outputs)
        """
        raise NotImplementedError
    
    def _apply_reward_modifications(self, obs, rewards, infos):
        """
        Apply environment-specific reward modifications (bonuses/penalties).
        
        Default: return rewards unchanged.
        Subclasses can override to add tracking bonuses, penalties, etc.
        
        Args:
            obs: Observations from gym.step()
            rewards: Rewards from gym.step()
            infos: Info dict from gym.step()
            
        Returns:
            Modified rewards array
        """
        return rewards
    
    @abstractmethod
    def _get_gym_actions(self):
        """
        Return the gymnasium actions buffer prepared by _process_agent_action().
        
        Subclass responsibility: return the action buffer in correct format for gym.step().
        
        Returns:
            Action array/buffer ready for gym.step()
        """
        raise NotImplementedError
    
    def _handle_render_mode(self, render_mode_raw):
        """
        Standardize render mode handling.
        
        Args:
            render_mode_raw: Raw render mode value (True/False/string/None)
            
        Returns:
            Standardized render mode (string or None)
        """
        if render_mode_raw is True:
            return "human"
        elif render_mode_raw is False:
            return None
        else:
            return render_mode_raw
    
    def _update_obs_cache(self, obs_array):
        """
        Update observation cache with latest observations.
        
        Default implementation normalizes observations from numpy arrays.
        Subclasses can override for custom normalization.
        
        Args:
            obs_array: Observation array from gymnasium environment
        """
        if isinstance(obs_array, np.ndarray):
            self.obs_cache.copy_(
                torch.from_numpy(obs_array.astype(np.float32)).to(self.device)
            )
        else:
            # Handle other array-like types
            self.obs_cache.copy_(
                torch.from_numpy(np.asarray(obs_array, dtype=np.float32)).to(self.device)
            )
    
    def _finalize_step(self, obs, rewards, terminateds, truncateds):
        """
        Update internal caches after gymnasium environment step.
        
        Common pattern after gym.step():
        - Update observation cache
        - Convert and store rewards
        - Compute and store done flags
        
        Args:
            obs: Observations from gym.step()
            rewards: Rewards from gym.step()
            terminateds: Terminal flags from gym.step()
            truncateds: Truncated flags from gym.step()
        """
        self._update_obs_cache(obs)
        
        self.current_rewards.copy_(
            torch.from_numpy(rewards.astype(np.float32)).to(self.device)
        )
        
        dones = terminateds | truncateds
        self.dones.copy_(
            torch.from_numpy(dones).to(self.device)
        )
    
    def reward(self, agent: Agent):
        """
        Return current rewards from environment step.
        
        Args:
            agent: VMAS Agent (unused, for BaseScenario compatibility)
            
        Returns:
            Tensor of shape (num_envs,) with current rewards
        """
        return self.current_rewards
    
    def observation(self, agent: Agent):
        """
        Return current observations.
        
        Args:
            agent: VMAS Agent (unused, for BaseScenario compatibility)
            
        Returns:
            Tensor of shape (num_envs, obs_dim) with observations
        """
        return self.obs_cache
    
    def done(self):
        """
        Return done flags for all environments.
        
        Returns:
            Boolean tensor of shape (num_envs,) indicating terminal states
        """
        return self.dones
    
    @abstractmethod
    def max_rewards(self):
        """
        Return maximum possible rewards for scaling/normalization.
        
        Must be implemented by subclasses since max reward is environment-specific.
        
        Returns:
            Tensor of shape (num_envs,) with maximum reward per environment
        """
        raise NotImplementedError
    
    def diffreward(self, prevs, acts, nexts):
        """
        Compute differentiable rewards (not applicable for non-differentiable environments).
        
        Raises:
            NotImplementedError: Gymnasium environments provide non-differentiable rewards
        """
        raise NotImplementedError("Gymnasium environments have non-differentiable rewards")
    
    def zero_grad(self):
        """
        Clear gradients (no-op for non-differentiable environments).
        
        This method is a no-op since gymnasium environments don't support
        automatic differentiation through the environment step.
        """
        pass
