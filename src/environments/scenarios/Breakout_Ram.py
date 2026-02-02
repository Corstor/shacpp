"""
Breakout_Ram Scenario - A VMAS-compatible wrapper using Atari RAM state.

This module provides a scenario that integrates Atari Breakout with the VMAS
framework using RAM observations instead of pixel-based CNN features.

Key Differences from Breakout.py (CNN version):
    - Uses ALE/Breakout-ram-v5 which provides 128-byte RAM state
    - No CNN preprocessing - direct RAM state as observations
    - Simpler and faster (no CNN forward pass)
    - RAM contains complete game state (ball position, velocity, bricks, etc.)
    - observation_size = 128 (instead of 512 for CNN)

"""

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import torch
import numpy as np
from vmas.simulator.core import Agent, World, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color
import ale_py

class FireResetEnv(gym.Wrapper):
    """
    Gymnasium wrapper that automatically fires at the start of each episode/life.
    
    In Breakout, the game requires a FIRE action (action=1) to launch the ball
    at the beginning of the game and after losing a life. This wrapper handles
    that automatically so the agent doesn't need to learn this trivial action.
    """
    
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        self.lives = 0
        self.was_real_done = True
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get('lives', 0)
        self.was_real_done = True

        # Fire to launch the ball
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        
        self.lives = info.get('lives', 0)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.was_real_done = done

        lives = info.get('lives', 0)
        if lives < self.lives and lives > 0:
            # Lost a life but game continues - fire to launch new ball
            self.was_real_done = False
            obs, _, _, _, info = self.env.step(1)
            self.lives = lives
        
        return obs, reward, terminated, truncated, info


class FrameSkipEnv(gym.Wrapper):
    """
    Skip frames and repeat action, summing rewards.
    
    This is needed for RAM version since AtariPreprocessing is for pixels.
    """
    
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        
    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
                
        return obs, total_reward, done, truncated, info


class Scenario(BaseScenario):
    """
    VMAS Scenario wrapper for Atari Breakout using RAM observations.
    
    This scenario uses the 128-byte RAM state instead of pixel observations.
    The RAM contains all game state information including:
    - Ball position and velocity
    - Paddle position
    - Brick states
    - Score and lives
    
    This may be easier for a world model to learn than pixel-based features
    since the state is more "physics-like" and lower dimensional.
    
    Attributes:
        observation_size: 128 (RAM bytes)
        action_size: 1 (continuous action discretized to LEFT/RIGHT/NOOP)
        agents: 1 (single paddle)
    """
    
    def make_world(self, num_envs: int, device: torch.device, **kwargs) -> World:
        """
        Create and initialize the VMAS world with vectorized Breakout RAM environments.
        
        Args:
            num_envs: Number of parallel environments
            device: Device for tensor computations
            **kwargs:
                - render_mode: Rendering mode (None, "human")
                - life_loss_penalty: Negative reward for losing a life (default 0.0)
                - frame_skip: Number of frames to skip per action (default 4)
                - async_vectorize: Use async environments (default: num_envs >= 16)
                - action_temperature: Temperature for action discretization (default 0.5, range = [0, 1], higher = more random)
        """
        self.num_envs = num_envs
        self.device = device
        self.observation_dim = 128  # RAM size
        
        # Configuration
        self.life_loss_penalty = kwargs.get("life_loss_penalty", 0.0)
        self.frame_skip = kwargs.get("frame_skip", 4)
        self.tracking_bonus = kwargs.get("tracking_bonus", 0.0)  # Dense reward for paddle tracking ball
        self.action_temperature = kwargs.get("action_temperature", 0.5)  # Softmax temperature for action discretization
        
        if self.life_loss_penalty > 0:
            print(f"✓ Life loss penalty enabled: -{self.life_loss_penalty} reward per life lost")
        if self.tracking_bonus > 0:
            print(f"✓ Tracking bonus enabled: +{self.tracking_bonus} max reward for paddle near ball")
        print(f"✓ Action temperature: {self.action_temperature} (lower = more deterministic)")
        
        # Register Atari environments
        gym.register_envs(ale_py)

        # Handle render mode
        render_mode_raw = kwargs.get("render_mode", None)
        if render_mode_raw is True:
            self.render_mode = "human"
        elif render_mode_raw is False:
            self.render_mode = None
        else:
            self.render_mode = render_mode_raw

        def make_env(render_mode=None):
            """Create a single Breakout RAM environment."""
            env = gym.make(
                "ALE/Breakout-v5",
                obs_type="ram",  # Use RAM observations (128 bytes)
                repeat_action_probability=0.0,
                frameskip=1,
                render_mode=render_mode
            )
            env = FrameSkipEnv(env, skip=self.frame_skip)
            env = FireResetEnv(env)
            return env
        
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
        
        # RAM observation cache: (num_envs, 128) normalized to [0, 1]
        self.obs_cache = torch.zeros(
            num_envs, self.observation_dim,
            device=device,
            dtype=torch.float32
        )
        
        # Pre-allocated buffers
        self._actions_buffer = np.zeros(num_envs, dtype=np.float32)
        self._gym_actions_buffer = np.zeros(num_envs, dtype=np.int32)
        self._lives_lost_buffer = np.zeros(num_envs, dtype=np.float32)
        self._tracking_distance_buffer = np.zeros(num_envs, dtype=np.float32)  # For tracking bonus

        # Create VMAS world (placeholder for interface compliance)
        world = World(
            batch_dim=num_envs,
            device=device,
            x_semidim=1,
            y_semidim=1,
            collision_force=0,
            substeps=1,
        )

        agent = Agent(
            name="paddle",
            shape=Sphere(radius=0.05),
            color=Color.BLUE,
            render_action=True
        )
        world.add_agent(agent)
        
        print(f"✓ BreakoutRAM initialized: {num_envs} envs, {self.observation_dim}-dim RAM observations")
        
        return world
    
    def reset_world_at(self, env_index: int = None):
        """Reset environment(s) to initial state."""
        obs, infos = self.gym_env.reset()
        self._update_obs_cache(obs)
        self.current_rewards.zero_()
        self.dones.zero_()
        self._prev_lives = np.full(self.num_envs, 5, dtype=np.int32)
    
    def _update_obs_cache(self, obs_array):
        """
        Update observation cache with RAM state.
        
        Normalizes RAM bytes from [0, 255] to [0, 1].
        """
        if isinstance(obs_array, np.ndarray):
            self.obs_cache.copy_(
                torch.from_numpy(obs_array.astype(np.float32) / 255.0).to(self.device)
            )
        else:
            for i, obs in enumerate(obs_array):
                self.obs_cache[i].copy_(
                    torch.from_numpy(np.array(obs, dtype=np.float32) / 255.0).to(self.device)
                )
    
    def reward(self, agent: Agent):
        """Return current rewards."""
        return self.current_rewards
    
    def observation(self, agent: Agent):
        """
        Return RAM observations.
        
        Returns:
            Tensor of shape (num_envs, 128) with normalized RAM state
        """
        return self.obs_cache
    
    def done(self):
        """Return done flags."""
        return self.dones
    
    def process_action(self, agent: Agent):
        """
        Convert continuous actions to discrete Breakout actions and step.
        
        Uses softmax-based stochastic discretization:
            - Continuous action [-1, 1] is mapped to logits for [NOOP, RIGHT, LEFT]
            - Temperature controls exploration (lower = more deterministic)
            - This provides smoother gradients for learning and balanced exploration
        """
        action_values = agent.action.u[:, 0].cpu().numpy()
        
        # Softmax-based stochastic discretization (vectorized)
        # Create logits: NOOP peaks at 0, RIGHT peaks at +1, LEFT peaks at -1
        logits = np.empty((self.num_envs, 3), dtype=np.float32)
        logits[:, 0] = -np.abs(action_values) * 2  # NOOP
        logits[:, 1] = action_values * 2           # RIGHT
        logits[:, 2] = -action_values * 2          # LEFT
        
        # Softmax with temperature (numerically stable)
        logits *= (1.0 / self.action_temperature)
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=1, keepdims=True)
        
        # Vectorized categorical sampling using cumsum + searchsorted
        cumprobs = np.cumsum(logits, axis=1)
        rand = np.random.random(self.num_envs).astype(np.float32)
        sampled_indices = np.sum(cumprobs < rand[:, None], axis=1)
        
        # Map indices [0,1,2] -> Atari actions [NOOP=0, RIGHT=2, LEFT=3]
        self._gym_actions_buffer[:] = np.take(np.array([0, 2, 3], dtype=np.int32), sampled_indices)

        # Step environments
        obs, rewards, terminateds, truncateds, infos = self.gym_env.step(self._gym_actions_buffer)

        # Apply tracking bonus: reward paddle for being close to ball X position
        # Atari 2600 Breakout RAM layout:
        #   - Paddle X position: RAM[72] (range ~40-168)
        #   - Ball X position: RAM[99] (range ~0-160)
        if self.tracking_bonus > 0:
            # Optimized: use pre-allocated buffer and in-place operations
            np.subtract(obs[:, 72], obs[:, 99], out=self._tracking_distance_buffer)
            np.abs(self._tracking_distance_buffer, out=self._tracking_distance_buffer)
            np.divide(self._tracking_distance_buffer, 80.0, out=self._tracking_distance_buffer)
            np.clip(self._tracking_distance_buffer, 0.0, 1.0, out=self._tracking_distance_buffer)
            # tracking_reward = bonus * (1 - normalized_distance)
            rewards = rewards + self.tracking_bonus * (1.0 - self._tracking_distance_buffer)

        # Apply life loss penalty
        if self.life_loss_penalty > 0 and hasattr(self, '_prev_lives'):
            if 'lives' in infos:
                current_lives = infos['lives']
            else:
                current_lives = self._prev_lives.copy()
                final_info = infos.get('final_info', [])
                for i, fi in enumerate(final_info):
                    if fi is not None and 'lives' in fi:
                        current_lives[i] = fi['lives']
            
            np.subtract(self._prev_lives, current_lives, out=self._lives_lost_buffer)
            np.clip(self._lives_lost_buffer, 0, 5, out=self._lives_lost_buffer)
            rewards = rewards - self._lives_lost_buffer * self.life_loss_penalty
            np.copyto(self._prev_lives, current_lives)

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
        """Maximum possible reward (all bricks cleared)."""
        return torch.full((self.num_envs,), 864.0, device=self.device)
    
    def diffreward(self, prevs, acts, nexts):
        """Not implemented - Breakout rewards are not differentiable."""
        raise NotImplementedError("Breakout has non-differentiable rewards")
    
    def zero_grad(self):
        """Clear gradients (no-op for non-differentiable env)."""
        pass
