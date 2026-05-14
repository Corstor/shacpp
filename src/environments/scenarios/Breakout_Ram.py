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
    
    Action Mode:
        Uses 2 continuous action outputs with argmax selection:
        - action[0]: LEFT (Atari action 3)
        - action[1]: RIGHT (Atari action 2)
        The discrete action is selected as argmax(action_values).
    
    Attributes:
        observation_size: 128 (RAM bytes)
        action_size: 2 (two continuous logits, argmax selected)
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
                - tracking_bonus: Dense reward for paddle near ball (default 0.0)
                - direction_bonus: Dense reward for moving toward ball (default 0.0)
        """
        self.num_envs = num_envs
        self.device = device
        self.observation_dim = 128  # RAM size
        
        # Configuration
        self.life_loss_penalty = kwargs.get("life_loss_penalty", 0.0)
        self.frame_skip = kwargs.get("frame_skip", 4)
        self.tracking_bonus = kwargs.get("tracking_bonus", 0.0)  # Dense reward for paddle tracking ball
        self.direction_bonus = kwargs.get("direction_bonus", 0.0)  # Dense reward for moving toward ball
        
        if self.life_loss_penalty > 0:
            print(f"✓ Life loss penalty enabled: -{self.life_loss_penalty} reward per life lost")
        if self.tracking_bonus > 0:
            print(f"✓ Tracking bonus enabled: +{self.tracking_bonus} max reward for paddle near ball")
        if self.direction_bonus > 0:
            print(f"✓ Direction bonus enabled: +{self.direction_bonus} reward for moving toward ball")
        print(f"✓ Action mode: 2 continuous actions (argmax selection)")
        print(f"✓ Action mapping: action[0]=LEFT, action[1]=RIGHT")
        print(f"  - Policy network should output 2 continuous dimensions per agent")
        
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
        self._gym_actions_buffer = np.zeros(num_envs, dtype=np.int32)
        self._lives_lost_buffer = np.zeros(num_envs, dtype=np.float32)
        self._tracking_distance_buffer = np.zeros(num_envs, dtype=np.float32)  # For tracking bonus
        self._ball_position_diff = np.zeros(num_envs, dtype=np.float32)  # For interception bonus
        self._direction_reward_buffer = np.zeros(num_envs, dtype=np.float32)  # For direction bonus
        self._paddle_direction_buffer = np.zeros(num_envs, dtype=np.float32)  # For direction bonus

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
        action_size_param = kwargs.get("action_size", 2)  # Default to 2 if not specified
        agent = Agent(
            name="paddle",
            shape=Sphere(radius=0.05),
            color=Color.BLUE,
            movable=False,      # Skip force/velocity integration
            rotatable=False,    # Skip torque/angular integration
            collide=False,      # Skip collision detection
            action_size=action_size_param,  # CRITICAL: tells VMAS we have N continuous actions
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
        Convert 2 continuous actions to discrete Breakout actions via argmax selection.
        
        Action mapping:
            - agent.action.u[:, 0]: LEFT action (Atari action 3)
            - agent.action.u[:, 1]: RIGHT action (Atari action 2)
        
        Selection: argmax(action_values) determines which discrete action is executed.
        This allows the policy to output multiple action logits and let the argmax select.
        Optimized: argmax is computed on GPU, only final actions converted to numpy.
        """
        # agent.action.u is (num_envs, 2), two continuous actions in [-1, 1]
        # Keep on GPU for argmax operation
        indices = torch.argmax(agent.action.u, dim=1)
        
        # Convert indices to numpy only for gymnasium compatibility
        indices_np = indices.cpu().numpy().astype(np.int32)
        
        # Map indices [0, 1, 2] -> Atari actions [LEFT=3, RIGHT=2]
        self._gym_actions_buffer[:] = np.take(np.array([3, 2], dtype=np.int32), indices_np)

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
            np.clip(self._tracking_distance_buffer, 0.0, 2.0, out=self._tracking_distance_buffer)
            # tracking_reward = bonus * (1 - normalized_distance)
            rewards = rewards + self.tracking_bonus * (1.0 - self._tracking_distance_buffer)

        # Apply interception bonus: reward paddle for moving TOWARD the ball position
        # This is better than "same direction as ball" because it tells policy to INTERCEPT
        if self.direction_bonus > 0:
            # Compute ball position relative to paddle: positive = ball is to the RIGHT
            # ball_x - paddle_x: >0 means ball is right of paddle, <0 means ball is left
            np.subtract(obs[:, 99].astype(np.float32), obs[:, 72].astype(np.float32), out=self._ball_position_diff)
            
            # Get paddle action direction from gym_actions_buffer
            # Atari actions: 0=NOOP, 2=RIGHT, 3=LEFT
            # Map to: NOOP->0, RIGHT->+1, LEFT->-1
            # Use direct comparison (vectorized, no allocation)
            self._paddle_direction_buffer[:] = 0.0
            self._paddle_direction_buffer[self._gym_actions_buffer == 2] = 1.0   # RIGHT
            self._paddle_direction_buffer[self._gym_actions_buffer == 3] = -1.0  # LEFT
            
            # Reward when paddle moves TOWARD the ball
            # If ball is to the right (positive) and paddle goes right (+1) -> reward
            # If ball is to the left (negative) and paddle goes left (-1) -> reward (neg * neg = pos)
            np.sign(self._ball_position_diff, out=self._direction_reward_buffer)
            np.multiply(self._direction_reward_buffer, self._paddle_direction_buffer, out=self._direction_reward_buffer)
            # +1 for moving toward ball, -1 for moving away, 0 for NOOP or ball directly above
            rewards = rewards + self.direction_bonus * self._direction_reward_buffer

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
