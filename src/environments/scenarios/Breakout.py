"""
Breakout Scenario - A VMAS-compatible wrapper for the Atari Breakout environment.

This module provides a scenario that integrates the classic Atari Breakout game
with the VMAS (Vectorized Multi-Agent Simulator) framework. It is designed to work
with the SHAC-WM (Short Horizon Actor Critic with World Model) algorithm using
transformer-based policy networks.

Key Design Decisions:
    - Uses a pre-trained CNN feature extractor from Stable-Baselines3 Zoo to convert
      raw 84x84 grayscale frames into a compact feature vector (default 512-dim).
    - The policy network receives these features as observations instead of raw pixels,
      making it easier for the transformer to learn meaningful representations.
    - Continuous actions from the policy are discretized into 3 Breakout actions:
      NOOP (0), RIGHT (2), LEFT (3).
    - Single agent scenario (the paddle) compatible with multi-agent VMAS interface.

Usage with shacwm:
    The environment is created via environments.get_environment("breakout", ...)
    which calls vmas.make_env with this scenario. The transformer policy will
    receive observations of shape (batch, agents=1, feature_dim=512).
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
    
    It also tracks lives to detect when the agent loses a life (but game continues)
    vs when the episode truly ends. The life_lost flag can be used to apply
    negative rewards for losing lives.
    """
    
    def __init__(self, env):
        super().__init__(env)
        # Verify the environment has FIRE as action 1
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        self.lives = 0
        self.was_real_done = True
        self.life_lost = False  # Flag to track if a life was lost this step
        
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
        self.life_lost = False

        # Fire to launch the ball
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            # If firing caused termination (rare edge case), reset again
            obs, info = self.env.reset(**kwargs)
        
        self.lives = info.get('lives', 0)
        return obs, info
    
    def step(self, action):
        """
        Execute action and handle life loss by automatically firing.
        
        When the agent loses a life but still has lives remaining,
        this wrapper automatically fires to continue the game.
        The life_lost flag is set to True when a life is lost.
        
        Args:
            action: The action to execute
            
        Returns:
            obs, reward, terminated, truncated, info: Standard gym step returns
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.was_real_done = done
        self.life_lost = False  # Reset flag each step

        lives = info.get('lives', 0)
        if lives < self.lives and lives > 0:
            # Lost a life but game continues - fire to launch new ball
            self.was_real_done = False
            self.life_lost = True  # Mark that we lost a life
            obs, _, _, _, info = self.env.step(1)
            self.lives = lives
        elif lives < self.lives and lives == 0:
            # Lost last life - game over
            self.life_lost = True
        
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
    
    def __init__(self, input_channels=4, feature_dim=128, device='cpu', pretrained=None):
        super().__init__()
        
        # Convolutional layers following DQN architecture
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        ).to(device)
        
        # After conv layers: 64 channels x 7 x 7 spatial size = 3136
        conv_output_size = 64 * 7 * 7
        
        # Fully connected layer to produce final features
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
            
            # Handle SB3 checkpoint format
            if 'policy' in state_dict:
                state_dict = self._extract_cnn_from_sb3(state_dict['policy'])
            
            self.load_state_dict(state_dict, strict=False)
            print(f"✓ Pretrained weights loaded from {pretrained_path_or_dict}")
        elif isinstance(pretrained_path_or_dict, dict):
            self.load_state_dict(pretrained_path_or_dict, strict=False)
            print("✓ Pretrained weights loaded from dictionary")
    
    def _extract_cnn_from_sb3(self, sb3_policy_dict):
        """
        Extract CNN weights from a Stable-Baselines3 policy dictionary.
        
        SB3 uses different key naming conventions, so this method maps
        the SB3 keys to our model's key names.
        
        Args:
            sb3_policy_dict: State dict from SB3 policy
            
        Returns:
            dict: Remapped state dict compatible with this model
        """
        cnn_dict = {}

        for key, value in sb3_policy_dict.items():
            if 'features_extractor.cnn' in key:
                # Map SB3 CNN layers to our conv layers
                new_key = key.replace('features_extractor.cnn.', 'conv.')
                cnn_dict[new_key] = value
            elif 'features_extractor.linear' in key:
                # Map SB3 linear layer to our fc layer
                new_key = key.replace('features_extractor.linear.', 'fc.')
                cnn_dict[new_key] = value
        
        return cnn_dict
    
    def freeze(self):
        """Freeze all CNN parameters to prevent gradient updates during training."""
        for param in self.parameters():
            param.requires_grad = False
        print("✓ CNN frozen - weights will not be updated")
    
    def unfreeze(self):
        """Unfreeze all CNN parameters to allow gradient updates during training."""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ CNN unfrozen - weights can be updated")
        
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
    
    The scenario is designed for use with shacwm + transformer, where:
        - observation_size = feature_dim (default 512)
        - action_size = 1 (single continuous action for paddle movement)
        - agents = 1 (single paddle agent)
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

            # Available pretrained models on HuggingFace
            available_models = {
                'sb3': 'sb3/dqn-BreakoutNoFrameskip-v4',
                'nsanghi-gpu': 'nsanghi/dqn-atari-breakout-rlzoo-gpu',
                'nsanghi': 'nsanghi/dqn-atari-breakout-rlzoo',
            }

            model_choice = 'sb3'
            repo_id = available_models[model_choice]
            
            print(f"📦 Downloading model from HuggingFace: {repo_id}")

            checkpoint_path = load_from_hub(
                repo_id=repo_id,
                filename=f"dqn-BreakoutNoFrameskip-v4.zip"
            )

            import zipfile
            import tempfile
            import os
            
            # Extract the zip file to access the model weights
            with tempfile.TemporaryDirectory() as tmpdirname:
                with zipfile.ZipFile(checkpoint_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdirname)

                # Try different possible file names for the model weights
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
                    print(f"✓ Pretrained CNN loaded from SB3 Zoo ({model_choice})!")
                    print(f"  Note: Model trained on v4, but CNN compatible with v5")
                    return cnn
                else:
                    print(f"⚠️  Model file not found in: {list(os.listdir(tmpdirname))}")
                    print("    Using CNN with random weights.")
                    return BreakoutCNN(
                        input_channels=4,
                        feature_dim=self.feature_dim,
                        device=device
                    )
        
        except ImportError:
            print("⚠️  huggingface_sb3 not installed. Install with:")
            print("    pip install huggingface-sb3")
            print("    Using CNN with random weights.")
            return BreakoutCNN(
                input_channels=4,
                feature_dim=self.feature_dim,
                device=device
            )
        except Exception as e:
            print(f"⚠️  Error loading SB3 model: {e}")
            print("    Using CNN with random weights.")
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
                - gym_env_name: Gym environment ID (default "ALE/Breakout-v5")
                - render_mode: Rendering mode (None, "human", True, False)
                - life_loss_penalty: Negative reward for losing a life (default 0.0)
                  Set to a positive value like 1.0 or 5.0 to penalize life loss.
                  This helps shape the reward and makes learning faster.
                
        Returns:
            World: VMAS World object (mostly placeholder, actual game state is in gym_env)
        """
        self.num_envs = num_envs
        self.device = device

        # Configuration from kwargs
        self.feature_dim = kwargs.get("feature_dim", 512)
        self.gym_env_name = kwargs.get("gym_env_name", "ALE/Breakout-v5")
        
        # Life loss penalty: negative reward when agent loses a life
        # This helps shape the reward signal and speeds up learning
        # Recommended values: 1.0 to 5.0 (or 0.0 to disable)
        self.life_loss_penalty = kwargs.get("life_loss_penalty", 0.0)
        if self.life_loss_penalty > 0:
            print(f"✓ Life loss penalty enabled: -{self.life_loss_penalty} reward per life lost")
        
        # Register Atari environments
        gym.register_envs(ale_py)

        # Handle different render_mode input formats
        render_mode_raw = kwargs.get("render_mode", None)
        if render_mode_raw is True:
            self.render_mode = "human"
        elif render_mode_raw is False:
            self.render_mode = None
        else:
            self.render_mode = render_mode_raw
        
        # Load pretrained CNN feature extractor
        self.cnn_encoder = self._load_sb3_zoo_cnn(self.device)
        # self.cnn_encoder.eval()  # Set to evaluation mode (no dropout, etc.)
        
        # NOTE: torch.compile disabled due to conflicts with deterministic mode
        # Uncomment if not using deterministic training:
        # if hasattr(torch, 'compile'):
        #     self.cnn_encoder = torch.compile(self.cnn_encoder, mode='reduce-overhead')

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
                self.gym_env_name,
                repeat_action_probability=0.0,  # Deterministic
                frameskip=1,  # We handle frameskip in AtariPreprocessing
                render_mode=render_mode
            )
            env = AtariPreprocessing(
                env,
                frame_skip=4,           # Skip 4 frames per action
                screen_size=84,         # Resize to 84x84
                terminal_on_life_loss=False,  # Don't end episode on life loss
                grayscale_obs=True,     # Convert to grayscale
                grayscale_newaxis=False,# Don't add channel dimension
                scale_obs=False,        # Keep as uint8, we scale later
            )
            env = FrameStackObservation(env, 4)  # Stack 4 frames
            env = FireResetEnv(env)  # Auto-fire at start/after life loss
            return env
        
        def make_env_with_render():
            """Create environment with rendering enabled."""
            return make_env(self.render_mode)

        # Create vectorized environments
        # Only the first environment renders if render_mode is "human"
        if self.render_mode == "human":
            env_fns = [make_env_with_render] + [make_env for _ in range(num_envs - 1)]
        else:
            env_fns = [make_env for _ in range(num_envs)]
        
        # OPTIMIZATION: Use SyncVectorEnv for small batch sizes (lower overhead)
        # AsyncVectorEnv has process spawning overhead that hurts small batches
        # Switch to async_vectorize=True for num_envs >= 16 if you have multicore CPU
        use_async = kwargs.get("async_vectorize", num_envs >= 16)
        if use_async:
            self.gym_env = AsyncVectorEnv(env_fns)
        else:
            from gymnasium.vector import SyncVectorEnv
            self.gym_env = SyncVectorEnv(env_fns)

        # Initialize state caches
        self.current_rewards = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.dones = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Observation cache: stores raw frames before CNN processing
        # Shape: (num_envs, 4 frames, 84 height, 84 width)
        self.obs_cache = torch.zeros(
            num_envs, 4, 84, 84, 
            device=device, 
            dtype=torch.float32
        )
        
        # ============== PRE-ALLOCATED BUFFERS FOR PERFORMANCE ==============
        # These buffers avoid repeated memory allocation during training
        
        # Action buffers: avoid repeated array creation in process_action
        self._actions_buffer = np.zeros(num_envs, dtype=np.float32)
        self._gym_actions_buffer = np.zeros(num_envs, dtype=np.int32)
        
        # Life loss tracking buffer (for life_loss_penalty feature)
        self._lives_lost_buffer = np.zeros(num_envs, dtype=np.float32)

        # Create minimal VMAS world (actual game logic is in gym_env)
        world = World(
            batch_dim=num_envs,
            device=device,
            x_semidim=1,
            y_semidim=1,
            collision_force=0,
            substeps=1,
        )

        # Add a single agent representing the paddle
        # This is mainly for VMAS interface compatibility
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
        
        Note: Currently resets ALL environments regardless of env_index.
        This is because AsyncVectorEnv doesn't support single-env reset easily.
        For shacwm training, this is acceptable since we reset all envs together.
        
        Args:
            env_index: Ignored (all environments are reset)
        """
        obs, infos = self.gym_env.reset()
        self._update_obs_cache(obs)
        self.current_rewards.zero_()
        self.dones.zero_()
        
        # Initialize lives tracking for life loss penalty
        # Breakout starts with 5 lives
        self._prev_lives = np.full(self.num_envs, 5, dtype=np.int32)
    
    def _update_obs_cache(self, obs_array):
        """
        Update the observation cache with new frames from gym environments.
        
        Converts numpy arrays to tensors and normalizes pixel values from
        [0, 255] to [0, 1] for CNN processing.
        
        Args:
            obs_array: Numpy array or list of observations from gym_env
        """
        if isinstance(obs_array, np.ndarray):
            # Convert numpy to tensor and normalize in one step
            self.obs_cache.copy_(
                torch.from_numpy(obs_array.astype(np.float32) / 255.0).to(self.device)
            )
        else:
            # Individual updates (rarely used)
            for i, obs in enumerate(obs_array):
                if hasattr(obs, '__array__'):
                    obs = np.array(obs)
                self.obs_cache[i].copy_(
                    torch.from_numpy(obs.astype(np.float32) / 255.0).to(self.device)
                )
    
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
        # OPTIMIZED: inference_mode is faster than no_grad (disables more tracking)
        with torch.inference_mode():
            return self.cnn_encoder(self.obs_cache)
    
    def reward(self, agent: Agent):
        """
        Return the current rewards for all environments.
        
        In Breakout, reward is the score gained in the last step
        (points from breaking bricks).
        
        Args:
            agent: VMAS agent (ignored, we only have one agent)
            
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
            agent: VMAS agent (ignored, we only have one agent)
            
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
        
        The transformer policy outputs continuous actions in [-1, 1].
        These are discretized as:
            - action < -0.33: LEFT (action 3)
            - action > 0.33: RIGHT (action 2)  
            - otherwise: NOOP (action 0)
        
        After stepping the gym environments, this method updates:
            - obs_cache with new frames
            - current_rewards with step rewards (minus life loss penalty if enabled)
            - dones with termination flags
        
        Args:
            agent: VMAS agent containing the action tensor
        """
        # Get continuous actions from VMAS agent
        # Shape: (num_envs, action_size=2)
        vmas_actions = agent.action.u

        # OPTIMIZED: Reuse pre-allocated buffer and avoid repeated numpy array creation
        # Extract the single action value per environment directly into buffer
        self._actions_buffer[:] = vmas_actions[:, 0].cpu().numpy()

        # Discretize continuous actions to Breakout discrete actions
        # Breakout actions: 0=NOOP, 2=RIGHT, 3=LEFT (1=FIRE is handled by wrapper)
        # OPTIMIZED: Use pre-allocated int array and in-place operations
        np.copyto(self._gym_actions_buffer, 0)  # Default NOOP
        self._gym_actions_buffer[self._actions_buffer < -0.33] = 3  # LEFT
        self._gym_actions_buffer[self._actions_buffer > 0.33] = 2   # RIGHT

        # Step all gym environments
        obs, rewards, terminateds, truncateds, infos = self.gym_env.step(self._gym_actions_buffer)

        # Apply life loss penalty if enabled
        # OPTIMIZED: Simplified life detection logic
        if self.life_loss_penalty > 0 and hasattr(self, '_prev_lives'):
            # Extract current lives efficiently
            # AsyncVectorEnv provides lives directly in infos
            if 'lives' in infos:
                current_lives = infos['lives']  # Already a numpy array
            else:
                # Fallback for final_info format
                current_lives = self._prev_lives.copy()
                final_info = infos.get('final_info', [])
                for i, fi in enumerate(final_info):
                    if fi is not None and 'lives' in fi:
                        current_lives[i] = fi['lives']
            
            # Detect and apply life loss penalty using pre-allocated buffer
            np.subtract(self._prev_lives, current_lives, out=self._lives_lost_buffer)
            np.clip(self._lives_lost_buffer, 0, 5, out=self._lives_lost_buffer)  # Clamp to valid range
            rewards = rewards - self._lives_lost_buffer * self.life_loss_penalty
            
            # Update previous lives
            np.copyto(self._prev_lives, current_lives)

        # Update caches with new state
        self._update_obs_cache(obs)
        
        # Update reward tensor
        self.current_rewards.copy_(
            torch.from_numpy(rewards.astype(np.float32)).to(self.device)
        )
        
        # Combine termination flags and update dones tensor
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


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=False,
        display_info=True,
    )
