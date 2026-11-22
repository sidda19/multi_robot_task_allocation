import sys
import os
import time
import json
import random
import numpy as np
import torch
from typing import Dict, List, Any

# Ensure libs in sys.path
_libs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "libs"))
if os.path.exists(_libs_path) and _libs_path not in sys.path:
    sys.path.insert(0, _libs_path)

from src.env.warehouse_env import WarehouseMRTAEnv
from src.communication.channel import CommunicationChannel
from src.marl.marl_agent import CooperativeMARLController
from config import DEFAULT_CONFIG, SimulationConfig


def train_idqn(
    n_episodes: int = 120,
    comm_level: str = "L2_Medium",
    save_dir: str = "models/idqn_model",
    config: SimulationConfig = DEFAULT_CONFIG,
    train_freq: int = 2,
    seed: int = 2026,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Trains Cooperative MARL (IDQN) agents in the warehouse environment.
    Lightweight, CPU-friendly training loop; actual runtime depends on hardware.
    """
    # Seed every RNG used by exploration, replay sampling, NumPy, and PyTorch.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    start_time = time.perf_counter()
    channel = CommunicationChannel(comm_level, config)
    env = WarehouseMRTAEnv(channel, config)

    obs_dim = env.obs_dim
    action_dim = config.n_tasks + 1

    controller = CooperativeMARLController(
        n_robots=config.n_robots,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=64,
        lr=1e-3,
        gamma=0.95,
        batch_size=32
    )

    episode_rewards = []
    episode_completions = []
    episode_conflicts = []

    if verbose:
        print(f"Starting IDQN Cooperative MARL Training for {n_episodes} episodes...", flush=True)
        print(f"Robots: {config.n_robots}, Tasks: {config.n_tasks}, Comm Level: {comm_level}", flush=True)

    step_counter = 0
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + ep * 17)
        masks = info["action_masks"]
        ep_reward = 0.0
        done = False
        ep_conflicts = 0

        while not done:
            step_counter += 1
            actions = controller.select_actions(obs, masks, training=True)
            next_obs, reward, done, step_info = env.step(actions)
            next_masks = step_info["action_masks"]

            controller.store_transitions(obs, actions, reward, next_obs, next_masks, done)

            if step_counter % train_freq == 0:
                controller.train_step()

            ep_reward += reward
            ep_conflicts += step_info.get("conflicts_this_step", 0)
            obs = next_obs
            masks = next_masks

        comp_rate = step_info.get("total_tasks_completed", 0) / float(config.n_tasks)
        episode_rewards.append(ep_reward)
        episode_completions.append(comp_rate)
        episode_conflicts.append(ep_conflicts)

        if verbose and (ep % 20 == 0 or ep == n_episodes):
            avg_rew = sum(episode_rewards[-20:]) / min(20, len(episode_rewards))
            avg_comp = (sum(episode_completions[-20:]) / min(20, len(episode_completions))) * 100
            eps = controller.agents[0].epsilon
            elapsed = time.perf_counter() - start_time
            print(f"Episode {ep:3d}/{n_episodes} | Avg Reward: {avg_rew:6.2f} | Avg Comp: {avg_comp:5.1f}% | Epsilon: {eps:.3f} | Elapsed: {elapsed:5.1f}s", flush=True)

    # Save trained model weights
    controller.save(save_dir)
    total_training_time = time.perf_counter() - start_time
    metadata = {
        "algorithm": "Independent DQN (IDQN)",
        "seed": seed,
        "episodes": n_episodes,
        "communication_level": comm_level,
        "train_frequency": train_freq,
        "observation_dimension": obs_dim,
        "action_dimension": action_dim,
        "hidden_dimension": 64,
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "batch_size": 32,
        "epsilon_initial": 1.0,
        "epsilon_minimum": 0.05,
        "epsilon_decay": 0.995,
        "training_time_seconds": total_training_time,
        "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else None,
        "episode_completion_mean": float(np.mean(episode_completions)) if episode_completions else None
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"Training successfully completed in {total_training_time:.2f} seconds!", flush=True)
        print(f"Model checkpoints saved to: {os.path.abspath(save_dir)}", flush=True)

    return {
        "controller": controller,
        "training_time_s": total_training_time,
        "episode_rewards": episode_rewards,
        "episode_completions": episode_completions,
        "episode_conflicts": episode_conflicts
    }


if __name__ == "__main__":
    train_idqn(n_episodes=120, seed=2026, verbose=True)
