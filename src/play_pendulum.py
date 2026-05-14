"""
Play Pendulum with a trained policy model.

Usage:
    python play_pendulum.py --policy-path <path_to_best.pkl>

Example:
    python play_pendulum.py --policy-path data/shacwm/pendulum/1/transformer/42/best.pkl
"""

import click
import torch
import json
import os
import time
import numpy as np

import models
import environments


@click.command()
@click.option("--policy-path", "policy_path", type=click.Path(exists=True), required=True,
              help="Path to the policy checkpoint file (best.pkl or models.pkl)")
@click.option("--device", "device", type=str, default="cpu",
              help="Device to run on (cpu recommended for visualization)")
@click.option("--episodes", "episodes", type=int, default=5,
              help="Number of episodes to play")
@click.option("--max-steps", "max_steps", type=int, default=200,
              help="Maximum steps per episode")
@click.option("--delay", "delay", type=float, default=0.02,
              help="Delay between frames in seconds (for visualization speed)")
@click.option("--deterministic/--stochastic", "deterministic", default=True,
              help="Use deterministic actions (no sampling) or stochastic")
def play(
    policy_path,
    device,
    episodes,
    max_steps,
    delay,
    deterministic,
):
    """
    Play Pendulum with a trained policy model.
    """
    print("=" * 60)
    print("PENDULUM PLAYER - Trained Model Visualization")
    print("=" * 60)
    
    # Try to load config from same directory
    checkpoint_dir = os.path.dirname(policy_path)
    config_path = os.path.join(checkpoint_dir, "locals.json")
    
    # Default parameters
    observation_size = 3  # [cos(theta), sin(theta), theta_dot]
    action_size = 2  # continuous torque
    hidden_size = 64
    feedforward_size = 128
    layers = 1
    heads = 1
    dropout = 0.0
    var = 1.0
    train_steps = 32
    policy_nn = "transformer"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded config from {config_path}")
        
        # Override parameters from config
        observation_size = config.get("observation_size", observation_size)
        action_size = config.get("action_size", action_size)
        hidden_size = config.get("policy_hidden_size", hidden_size)
        feedforward_size = config.get("policy_feedforward", feedforward_size)
        layers = config.get("policy_layers", layers)
        heads = config.get("policy_heads", heads)
        dropout = config.get("policy_dropout", dropout)
        var = config.get("policy_var", var)
        train_steps = config.get("train_steps", train_steps)
        policy_nn = config.get("policy_nn", policy_nn)
    else:
        print("⚠️  No config found, using default parameters")
    
    print(f"\nModel configuration:")
    print(f"  policy_nn: {policy_nn}")
    print(f"  observation_size: {observation_size}")
    print(f"  action_size: {action_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  layers: {layers}")
    
    # Create policy model based on policy_nn type
    print("\n📦 Loading policy model...")
    
    if policy_nn == "mlp":
        policy_model = models.policies.MLPAFO(
            observation_size=observation_size,
            action_size=action_size,
            agents=1,
            steps=train_steps,
            action_space=[-1, 1],
            layers=layers,
            hidden_size=hidden_size,
            dropout=dropout,
            activation="ReLU",
            device=device,
        )
    else:
        # Default to transformer
        policy_model = models.policies.Transformer(
            observation_size=observation_size,
            action_size=action_size,
            agents=1,
            steps=train_steps,
            action_space=[-1, 1],
            layers=layers,
            hidden_size=hidden_size,
            feedforward_size=feedforward_size,
            heads=heads,
            dropout=dropout,
            activation="ReLU",
            var=var,
            device=device,
        )
    
    # Load checkpoint
    checkpoint = torch.load(policy_path, map_location=device, weights_only=True)
    
    # Handle compiled model keys (remove _orig_mod. prefix)
    state_dict = checkpoint.get("policy_state_dict", checkpoint)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    policy_model.load_state_dict(state_dict)
    policy_model.eval()
    print(f"✓ Policy loaded from {policy_path}")
    
    # Create environment with rendering
    print(f"\n🎮 Creating Pendulum environment with rendering...")
    
    import vmas
    env = vmas.make_env(
        scenario=environments.scenarios.Pendulum(),
        num_envs=1,
        device=device,
        continuous_actions=True,
        seed=42,
        grad_enabled=False,
        render_mode="human",
    )
    
    print("✓ Environment created")
    print(f"\n{'=' * 60}")
    print(f"Starting {episodes} episode(s)... Press Ctrl+C to stop")
    print(f"{'=' * 60}\n")
    
    total_rewards = []
    
    try:
        for episode in range(episodes):
            # Reset environment
            obs_list = env.reset()
            observations = torch.stack(obs_list).transpose(0, 1).to(device)  # (1, 1, 3)
            
            episode_reward = 0
            step = 0
            done = False
            
            print(f"\n--- Episode {episode + 1}/{episodes} ---")
            
            while not done and step < max_steps:
                # Get action from policy
                with torch.no_grad():
                    if deterministic:
                        result = policy_model.act(observations)
                    else:
                        result = policy_model.sample(observations)
                    actions = result["actions"]
                
                # Debug first few steps - show continuous action
                if step < 10:
                    action_val = actions[0, 0, 0].item()
                    torque = action_val * 2.0  # Scaled to [-2, 2]
                    print(f"  Step {step}: action={action_val:.3f}, torque={torque:.3f}")
                
                # Step environment
                actions_list = [actions[:, i, :] for i in range(actions.shape[1])]
                obs_list, rewards, dones, info = env.step(actions_list)
                
                # Process observations
                observations = torch.stack(obs_list).transpose(0, 1).to(device)
                reward = sum(r.item() for r in rewards)
                episode_reward += reward
                
                # Check done
                done = dones.any().item()
                step += 1
                
                # Delay for visualization
                if delay > 0:
                    time.sleep(delay)
                
                # Print progress
                if step % 50 == 0:
                    print(f"  Step {step}: Total Reward = {episode_reward:.2f}")
            
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1} finished: Reward = {episode_reward:.2f}, Steps = {step}")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    
    finally:
        # Close environment
        env.scenario.gym_env.close()
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Episodes played: {len(total_rewards)}")
    if total_rewards:
        print(f"Average reward: {np.mean(total_rewards):.2f}")
        print(f"Max reward: {max(total_rewards):.2f}")
        print(f"Min reward: {min(total_rewards):.2f}")


if __name__ == "__main__":
    play()
