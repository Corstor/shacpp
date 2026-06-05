"""
Breakout Scenario - A VMAS-compatible wrapper for the Atari Breakout environment.

This module provides a scenario that integrates the classic Atari Breakout game
with the VMAS (Vectorized Multi-Agent Simulator) framework.

Key Design Decisions:
    - Uses a pre-trained CNN feature extractor from Stable-Baselines3 Zoo to convert
      raw 84x84 grayscale frames into a compact feature vector (default 512-dim).
    - The policy network receives these features as observations instead of raw pixels,
      making it easier for the agent to learn meaningful representations.
    - Continuous actions from the policy are discretized into 3 Breakout actions:
      NOOP (0), RIGHT (2), LEFT (3).
    - Single agent scenario (the paddle) compatible with multi-agent VMAS interface.
"""

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from gymnasium.vector import AsyncVectorEnv
import torch
import torch.nn as nn
import numpy as np
from vmas import render_interactively
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
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        """
        Reset the environment and automatically fire to start the game.

        Returns:
            obs: The initial observation after firing
            info: Environment info dictionary
        """
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get('lives', 0)
        self.was_real_done = True
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        self.lives = info.get('lives', 0)
        return obs, info

    def step(self, action):
        """
        Execute action and handle life loss by automatically firing.

        When the agent loses a life but still has lives remaining,
        this wrapper automatically fires to continue the game.

        Args:
            action: The action to execute

        Returns:
            obs, reward, terminated, truncated, info: Standard gym step returns
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.was_real_done = done

        lives = info.get('lives', 0)
        if lives < self.lives and lives > 0:
            self.was_real_done = False
            obs, _, _, _, info = self.env.step(1)
            self.lives = lives
        
        return obs, reward, terminated, truncated, info

class BreakoutCNN(nn.Module):
    """
    Convolutional Neural Network for extracting features from Breakout frames.

    This CNN follows the architecture used by DQN (Mnih et al., 2015) and
    Stable-Baselines3. It processes stacked grayscale frames (4x84x84) and
    outputs a feature vector that can be used by downstream policy networks.

    Architecture:
        - Conv2d(4, 32, 8x8, stride=4) -> ReLU
        - Conv2d(32, 64, 4x4, stride=2) -> ReLU  
        - Conv2d(64, 64, 3x3, stride=1) -> ReLU
        - Flatten -> Linear(3136, feature_dim) -> ReLU

    The CNN can be initialized with pretrained weights from Stable-Baselines3
    models, which significantly improves learning performance since the
    visual features are already well-learned.

    Args:
        input_channels: Number of input channels (4 for frame stacking)
        feature_dim: Output feature dimension (observation size for policy)
        device: Device to place the model on
        pretrained: Path to pretrained weights or weight dictionary
    """

    def __init__(self, input_channels=4, feature_dim=512, device='cpu', pretrained=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        ).to(device)

        with torch.inference_mode():
            dummy_input = torch.zeros(1, input_channels, 84, 84, device=device)
            conv_output = self.conv(dummy_input)
            conv_output_size = conv_output.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, feature_dim),
            nn.ReLU(inplace=True),
        ).to(device)

        self.device = device
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained_path_or_dict):
        """
        Load pretrained weights from a file path or dictionary.

        Supports both raw state dicts and Stable-Baselines3 checkpoint format.

        Args:
            pretrained_path_or_dict: File path string or weight dictionary
        """
        if isinstance(pretrained_path_or_dict, str):
            state_dict = torch.load(pretrained_path_or_dict, map_location=self.device)

            if 'policy' in state_dict:
                state_dict = self._extract_cnn_from_sb3(state_dict['policy'])
            self.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights loaded from {pretrained_path_or_dict}")
        elif isinstance(pretrained_path_or_dict, dict):
            self.load_state_dict(pretrained_path_or_dict, strict=False)
            print("Pretrained weights loaded from dictionary")
    
    def _extract_cnn_from_sb3(self, sb3_policy_dict):
        """
        Extract CNN weights from a Stable-Baselines3 policy dictionary.

        SB3 uses different key naming conventions, so this method maps
        the SB3 keys to the model's key names.

        Args:
            sb3_policy_dict: State dict from SB3 policy

        Returns:
            dict: Remapped state dict compatible with this model
        """
        cnn_dict = {}

        for key, value in sb3_policy_dict.items():
            if 'features_extractor.cnn' in key:
                new_key = key.replace('features_extractor.cnn.', 'conv.')
                cnn_dict[new_key] = value
            elif 'features_extractor.linear' in key:
                new_key = key.replace('features_extractor.linear.', 'fc.')
                cnn_dict[new_key] = value

        return cnn_dict

    def freeze(self):
        """Freeze all CNN parameters to prevent gradient updates during training."""
        for param in self.parameters():
            param.requires_grad = False
        print("CNN frozen - weights will not be updated")

    def unfreeze(self):
        """Unfreeze all CNN parameters to allow gradient updates during training."""
        for param in self.parameters():
            param.requires_grad = True
        print("CNN unfrozen - weights can be updated")

    def forward(self, x):
        """
        Extract features from input frames.

        Args:
            x: Input tensor of shape (batch, 4, 84, 84) with values in [0, 1]

        Returns:
            Feature tensor of shape (batch, feature_dim)
        """
        conv_out = self.conv(x)
        conv_flat = conv_out.view(x.size(0), -1)
        features = self.fc(conv_flat)
        return features

class Scenario(BaseScenario):
    """
    VMAS Scenario wrapper for Atari Breakout environment.

    This scenario bridges the gap between the VMAS multi-agent framework and
    the single-agent Atari Breakout game. It manages:

    1. **Vectorized Gym environments**: Multiple Breakout games run in parallel
       using AsyncVectorEnv for efficient batch processing.

    2. **CNN feature extraction**: A pretrained CNN converts raw 84x84 grayscale
       frames into compact feature vectors that serve as observations.

    3. **Action conversion**: Continuous actions from the policy (in [-1, 1])
       are discretized into Breakout's 3 discrete actions (NOOP, RIGHT, LEFT).

    4. **VMAS interface compliance**: Implements all required VMAS scenario
       methods (reset_world_at, observation, reward, done, process_action).
    """

    def _load_sb3_zoo_cnn(self, device):
        """
        Load pretrained CNN weights from Stable-Baselines3 model zoo via HuggingFace.

        This method attempts to download a DQN model trained on Breakout from
        the HuggingFace model hub. The CNN portion of this model provides
        excellent visual features without needing to train from scratch.

        Args:
            device: Device to place the CNN on

        Returns:
            BreakoutCNN: CNN model with pretrained weights (or random if loading fails)
        """
        try:
            from huggingface_sb3 import load_from_hub
            available_models = {
                'sb3': 'sb3/dqn-BreakoutNoFrameskip-v4',
                'nsanghi-gpu': 'nsanghi/dqn-atari-breakout-rlzoo-gpu',
                'nsanghi': 'nsanghi/dqn-atari-breakout-rlzoo',
            }
            model_choice = 'sb3'
            repo_id = available_models[model_choice]
            print(f"Downloading model from HuggingFace: {repo_id}")
            checkpoint_path = load_from_hub(
                repo_id=repo_id,
                filename=f"dqn-BreakoutNoFrameskip-v4.zip"
            )
            import zipfile
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as tmpdirname:
                with zipfile.ZipFile(checkpoint_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdirname)

                possible_paths = [
                    os.path.join(tmpdirname, "policy.pth"),
                    os.path.join(tmpdirname, "dqn-BreakoutNoFrameskip-v4.pth"),
                    os.path.join(tmpdirname, "model.pth"),
                ]
                model_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break

                if model_path:
                    cnn = BreakoutCNN(
                        input_channels=4,
                        feature_dim=self.feature_dim,
                        device=device,
                        pretrained=model_path
                    )
                    print(f"Pretrained CNN loaded from SB3 Zoo ({model_choice})")
                    return cnn
                else:
                    print(f"Model file not found in: {list(os.listdir(tmpdirname))}")
                    print("Using CNN with random weights.")
                    return BreakoutCNN(
                        input_channels=4,
                        feature_dim=self.feature_dim,
                        device=device
                    )

        except ImportError:
            print("huggingface_sb3 not installed. Install with:")
            print("pip install huggingface-sb3")
            print("Using CNN with random weights.")
            return BreakoutCNN(
                input_channels=4,
                feature_dim=self.feature_dim,
                device=device
            )
        except Exception as e:
            print(f"Error loading SB3 model: {e}")
            print("Using CNN with random weights.")
            return BreakoutCNN(
                input_channels=4,
                feature_dim=self.feature_dim,
                device=device
            )

    def make_world(self, num_envs: int, device: torch.device, **kwargs) -> World:
        """
        Create and initialize the VMAS world with vectorized Breakout environments.

        This method sets up:
        1. Multiple parallel Breakout gym environments (AsyncVectorEnv)
        2. The CNN feature extractor with pretrained weights
        3. Observation and reward caches for efficient tensor operations
        4. A dummy VMAS world with a single "paddle" agent

        Args:
            num_envs: Number of parallel environments to create
            device: Device for tensor computations
            **kwargs: Additional arguments:
                - feature_dim: CNN output dimension (default 512)
                - render_mode: Rendering mode (None, "human", True, False)

        Returns:
            World: VMAS World object (mostly placeholder, actual game state is in gym_env)
        """
        self.num_envs = num_envs
        self.device = device

        self.feature_dim = kwargs.get("feature_dim", 512)

        gym.register_envs(ale_py)

        render_mode_raw = kwargs.get("render_mode", None)
        if render_mode_raw is True:
            self.render_mode = "human"
        elif render_mode_raw is False:
            self.render_mode = None
        else:
            self.render_mode = render_mode_raw

        self.cnn_encoder = self._load_sb3_zoo_cnn(self.device)

        def make_env(render_mode=None):
            """
            Create a single Breakout environment with standard preprocessing.

            Preprocessing includes:
            - Frame skipping (4 frames)
            - Resizing to 84x84
            - Grayscale conversion
            - Frame stacking (4 frames)
            - Auto-fire wrapper
            """
            env = gym.make(
                "ALE/Breakout-v5",
                repeat_action_probability=0.0,
                frameskip=1,
                render_mode=render_mode
            )
            env = AtariPreprocessing(
                env,
                frame_skip=4,
                screen_size=84,
                terminal_on_life_loss=False,
                grayscale_obs=True,
                grayscale_newaxis=False,
                scale_obs=False,
            )
            env = FrameStackObservation(env, 4)
            env = FireResetEnv(env)
            return env

        def make_env_with_render():
            """Create environment with rendering enabled."""
            return make_env(self.render_mode)

        if self.render_mode == "human":
            env_fns = [make_env_with_render] + [make_env for _ in range(num_envs - 1)]
        else:
            env_fns = [make_env for _ in range(num_envs)]

        use_async = kwargs.get("async_vectorize", num_envs >= 16)
        if use_async:
            self.gym_env = AsyncVectorEnv(env_fns)
        else:
            from gymnasium.vector import SyncVectorEnv
            self.gym_env = SyncVectorEnv(env_fns)

        self.current_rewards = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.obs_cache = torch.zeros(
            num_envs, 4, 84, 84, 
            device=device, 
            dtype=torch.float32
        )

        self._actions_buffer = np.zeros(num_envs, dtype=np.float32)
        self._gym_actions_buffer = np.zeros(num_envs, dtype=np.int32)
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
            render_action=True,
        )
        world.add_agent(agent)
        return world
    
    def reset_world_at(self, env_index: int = None):
        """
        Reset the environment(s) to initial state.
        
        Args:
            env_index: Ignored (all environments are reset)
        """
        obs, infos = self.gym_env.reset()
        self._update_obs_cache(obs)
        self.current_rewards.zero_()
        self.dones.zero_()

    def _update_obs_cache(self, obs_array):
        """
        Update the observation cache with new frames from gym environments.

        Converts numpy arrays to tensors and normalizes pixel values from
        [0, 255] to [0, 1] for CNN processing.

        Args:
            obs_array: Numpy array or list of observations from gym_env
        """
        if isinstance(obs_array, np.ndarray):
            self.obs_cache.copy_(
                torch.from_numpy(obs_array.astype(np.float32) / 255.0).to(self.device)
            )
        else:
            for i, obs in enumerate(obs_array):
                self._update_single_obs(i, obs)

    def _update_single_obs(self, idx, obs):
        """
        Update observation cache for a single environment.

        Args:
            idx: Environment index
            obs: Observation array
        """
        if hasattr(obs, '__array__'):
            obs = np.array(obs)
        self.obs_cache[idx].copy_(
            torch.from_numpy(obs).float().div_(255.0)
        )

    def _extract_features_batch(self):
        """
        Extract CNN features from all cached observations.

        Returns:
            Tensor of shape (num_envs, feature_dim) containing extracted features
        """
        with torch.inference_mode():
            return self.cnn_encoder(self.obs_cache)
    
    def reward(self, agent: Agent):
        """
        Return the current rewards for all environments.

        In Breakout, reward is the score gained in the last step
        (points from breaking bricks).

        Args:
            agent: VMAS agent

        Returns:
            Tensor of shape (num_envs,) with current step rewards
        """
        return self.current_rewards

    def observation(self, agent: Agent):
        """
        Return the current observations (CNN features) for all environments.

        This extracts features from the cached raw frames using the CNN encoder.
        The transformer policy receives these features, not raw pixels.

        Args:
            agent: VMAS agent

        Returns:
            Tensor of shape (num_envs, feature_dim) with CNN features
        """
        return self._extract_features_batch()

    def done(self):
        """
        Return done flags for all environments.

        Returns:
            Boolean tensor of shape (num_envs,) indicating episode termination
        """
        return self.dones
    
    def process_action(self, agent: Agent):
        """
        Convert continuous policy actions to discrete Breakout actions and step.

        The agent policy outputs continuous actions in [-1, 1].
        These are discretized as:
            - action < -0.33: LEFT (action 3)
            - action > 0.33: RIGHT (action 2)  
            - otherwise: NOOP (action 0)

        After stepping the gym environments, this method updates:
            - obs_cache with new frames
            - current_rewards with step rewards
            - dones with termination flags

        Args:
            agent: VMAS agent containing the action tensor
        """
        vmas_actions = agent.action.u
        self._actions_buffer[:] = vmas_actions[:, 0].cpu().numpy()

        np.copyto(self._gym_actions_buffer, 0)  # Default NOOP
        self._gym_actions_buffer[self._actions_buffer < -0.33] = 3  # LEFT
        self._gym_actions_buffer[self._actions_buffer > 0.33] = 2   # RIGHT

        obs, rewards, terminateds, truncateds, infos = self.gym_env.step(self._gym_actions_buffer)
        self._update_obs_cache(obs)
        self.current_rewards.copy_(
            torch.from_numpy(rewards.astype(np.float32)).to(self.device)
        )
        dones = terminateds | truncateds
        self.dones.copy_(
            torch.from_numpy(dones).to(self.device)
        )

    def max_rewards(self):
        """
        Return the theoretical maximum reward for early stopping evaluation.

        In Breakout, the maximum possible score is 864 (all bricks cleared
        with maximum multiplier). This is used by shacwm for early stopping
        criteria evaluation.

        Returns:
            Tensor of shape (num_envs,) filled with 864.0
        """
        return torch.full((self.num_envs,), 864.0, device=self.device)
    
    def diffreward(self, prevs, acts, nexts):
        """
        Compute differentiable reward (not implemented for Breakout).

        Breakout rewards are inherently non-differentiable (discrete brick
        breaking events), so this method is not used. The shacwm algorithm
        uses use_diffreward=False by default for this scenario.

        Raises:
            NotImplementedError: Always, as Breakout rewards are not differentiable
        """
        raise NotImplementedError("Breakout has non-differentiable rewards")

    def zero_grad(self):
        """
        Clear gradients (required by VMAS interface).

        Since Breakout doesn't use differentiable simulation, this is a no-op.
        """
        pass
