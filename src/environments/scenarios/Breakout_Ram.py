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
import torch
import numpy as np
from vmas.simulator.core import Agent, World
from vmas.simulator.utils import Color
import ale_py
from .BaseGymScenario import BaseGymScenario

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

class Scenario(BaseGymScenario):
    """
    VMAS Scenario wrapper for Atari Breakout using RAM observations.

    This scenario uses the 128-byte RAM state of the game.
    The RAM contains all game state information including:
    - Ball position and velocity
    - Paddle position
    - Brick states
    - Score and lives

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

    def _get_observation_dim(self) -> int:
        """Return observation dimension for Breakout: 128 RAM bytes"""
        return 128

    def _create_gym_env(self, render_mode):
        """Create a single Breakout RAM environment with wrappers."""
        gym.register_envs(ale_py)
        env = gym.make(
            "ALE/Breakout-v5",
            obs_type="ram",
            repeat_action_probability=0.0,
            frameskip=1,
            render_mode=render_mode
        )
        env = FrameSkipEnv(env, skip=self.frame_skip)
        env = FireResetEnv(env)
        return env

    def _get_agent_config(self) -> tuple:
        """Return (name, color) for Breakout agent."""
        return ("paddle", Color.BLUE)

    def _initialize_buffers(self, num_envs: int):
        """Initialize pre-allocated buffers for Breakout reward modifications and actions."""
        self._gym_actions_buffer = np.zeros(num_envs, dtype=np.int32)
        self._tracking_distance_buffer = np.zeros(num_envs, dtype=np.float32)
        self._ball_position_diff = np.zeros(num_envs, dtype=np.float32)
        self._direction_reward_buffer = np.zeros(num_envs, dtype=np.float32)
        self._paddle_direction_buffer = np.zeros(num_envs, dtype=np.float32)
    
    def make_world(self, num_envs: int, device: torch.device, **kwargs) -> World:
        """
        Override to handle Breakout-specific configuration before calling parent.
        """
        self.frame_skip = kwargs.get("frame_skip", 4)
        self.tracking_bonus = kwargs.get("tracking_bonus", 0.0)
        self.direction_bonus = kwargs.get("direction_bonus", 0.0)
        world = super().make_world(num_envs, device, **kwargs)

        if self.tracking_bonus > 0:
            print(f"Tracking bonus enabled: +{self.tracking_bonus} max reward for paddle near ball")
        if self.direction_bonus > 0:
            print(f"Direction bonus enabled: +{self.direction_bonus} reward for moving toward ball")
        return world

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

    def _process_agent_action(self, agent: Agent):
        """Convert 2 continuous actions to discrete via argmax."""
        indices = torch.argmax(agent.action.u, dim=1)
        indices_np = indices.cpu().numpy().astype(np.int32)
        self._gym_actions_buffer[:] = np.take(np.array([3, 2], dtype=np.int32), indices_np)

    def _get_gym_actions(self):
        """Return action buffer for gym.step()."""
        return self._gym_actions_buffer

    def _apply_reward_modifications(self, obs, rewards, infos):
        """
        Apply tracking bonus and direction bonus rewards if enabled.

        Tracking bonus: reward paddle for being close to ball X position
        Atari 2600 Breakout RAM layout:
            - Paddle X position: RAM[72] (range ~40-168)
            - Ball X position: RAM[99] (range ~0-160)

        Interception bonus: reward paddle for moving toward the ball position
        """
        if self.tracking_bonus > 0:
            np.subtract(obs[:, 72], obs[:, 99], out=self._tracking_distance_buffer)
            np.abs(self._tracking_distance_buffer, out=self._tracking_distance_buffer)
            np.divide(self._tracking_distance_buffer, 80.0, out=self._tracking_distance_buffer)
            np.clip(self._tracking_distance_buffer, 0.0, 1.0, out=self._tracking_distance_buffer)
            rewards = rewards + self.tracking_bonus * (1.0 - self._tracking_distance_buffer)

        if self.direction_bonus > 0:
            np.subtract(obs[:, 99].astype(np.float32), obs[:, 72].astype(np.float32), out=self._ball_position_diff)
            self._paddle_direction_buffer[:] = 1.0 # RIGHT
            self._paddle_direction_buffer[self._gym_actions_buffer == 3] = -1.0  # LEFT
            np.sign(self._ball_position_diff, out=self._direction_reward_buffer)
            np.multiply(self._direction_reward_buffer, self._paddle_direction_buffer, out=self._direction_reward_buffer)
            rewards = rewards + self.direction_bonus * self._direction_reward_buffer

        return rewards

    def max_rewards(self):
        """Maximum possible reward (all bricks cleared). Does not count in the tracking and interception bonuses"""
        return torch.full((self.num_envs,), 864.0, device=self.device)
