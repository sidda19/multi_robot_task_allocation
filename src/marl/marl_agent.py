import sys
import os
import random
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Automatically include local libs if present
_libs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "libs"))
if os.path.exists(_libs_path) and _libs_path not in sys.path:
    sys.path.insert(0, _libs_path)

import torch
import torch.nn as nn
import torch.optim as optim

from src.marl.q_network import MaskedQNetwork


class ReplayBuffer:
    """
    Fixed-size experience replay buffer for storing multi-agent transitions.
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, next_mask: np.ndarray, done: bool):
        self.buffer.append((
            np.array(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            np.array(next_mask, dtype=bool),
            bool(done)
        ))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transitions = random.sample(self.buffer, batch_size)
        obs_b, act_b, rew_b, next_obs_b, next_mask_b, done_b = zip(*transitions)

        return (
            torch.from_numpy(np.stack(obs_b)),
            torch.from_numpy(np.array(act_b, dtype=np.int64)).unsqueeze(1),
            torch.from_numpy(np.array(rew_b, dtype=np.float32)).unsqueeze(1),
            torch.from_numpy(np.stack(next_obs_b)),
            torch.from_numpy(np.stack(next_mask_b)),
            torch.from_numpy(np.array(done_b, dtype=np.float32)).unsqueeze(1)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class IDQNAgent:
    """
    Independent Deep Q-Network (IDQN) Agent.
    Operates with:
    - 2-layer MLP (hidden_dim=64)
    - Action masking during both exploration and exploitation
    - Target network synchronization
    - Experience replay buffer
    - Epsilon-greedy exploration with decay
    """
    def __init__(
        self,
        agent_id: int = 0,
        obs_dim: int = 87,
        action_dim: int = 11,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_capacity: int = 10000,
        target_update_freq: int = 50,
        device: str = "cpu"
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        # Primary and Target Q-networks
        self.q_network = MaskedQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = MaskedQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, training: bool = True) -> int:
        """
        Selects an action respecting the boolean action mask.
        In exploration (training with probability epsilon), uniformly samples among valid actions.
        In exploitation, selects argmax over masked Q-values.
        """
        mask = np.array(action_mask, dtype=bool)
        valid_actions = np.where(mask)[0]

        # Safety fallback: WAIT action (last action) is always available
        if len(valid_actions) == 0:
            return self.action_dim - 1

        # Epsilon-greedy exploration: sample strictly from VALID actions
        if training and (random.random() < self.epsilon):
            return int(np.random.choice(valid_actions))

        # Exploitation via Q-network
        self.q_network.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(np.array(observation, dtype=np.float32)).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            q_vals = self.q_network(obs_t, mask_t)
            action = int(torch.argmax(q_vals, dim=-1).item())

        self.q_network.train()

        # Final validity check
        if not mask[action]:
            action = int(valid_actions[0])

        return action

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_mask: np.ndarray,
        done: bool
    ):
        """Stores transition in experience replay buffer."""
        self.replay_buffer.push(obs, action, reward, next_obs, next_mask, done)

    def train_step(self) -> Optional[float]:
        """
        Performs one Bellman Q-learning update step.
        Returns loss value or None if buffer has insufficient samples.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, next_masks, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_masks = next_masks.to(self.device)
        dones = dones.to(self.device)

        # Q(s, a)
        current_q = self.q_network(states).gather(1, actions)

        # Target Q = r + gamma * max_a' Q_target(s', a') with action masking
        with torch.no_grad():
            next_q_vals = self.target_network(next_states, mask=next_masks)
            max_next_q = next_q_vals.max(1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def save(self, filepath: str):
        """Saves agent weights and hyperparameters."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done
        }, filepath)

    def load(self, filepath: str):
        """Loads agent weights and hyperparameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        if "steps_done" in checkpoint:
            self.steps_done = checkpoint["steps_done"]


class CooperativeMARLController:
    """
    Cooperative Multi-Agent Reinforcement Learning Controller.
    Coordinates N heterogeneous IDQN agents optimizing shared team reward.
    Provides both multi-agent and single-agent interface.
    """
    def __init__(
        self,
        n_robots: int = 3,
        obs_dim: int = 87,
        action_dim: int = 11,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.95,
        batch_size: int = 32
    ):
        self.n_robots = n_robots
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agents: Dict[int, IDQNAgent] = {
            i: IDQNAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lr=lr,
                gamma=gamma,
                batch_size=batch_size
            )
            for i in range(n_robots)
        }

    def select_actions(
        self,
        observations: Dict[int, np.ndarray],
        action_masks: Dict[int, np.ndarray],
        training: bool = True
    ) -> Dict[int, int]:
        """Selects actions for all robots."""
        actions = {}
        for r_id, agent in self.agents.items():
            obs = observations[r_id]
            mask = action_masks[r_id]
            actions[r_id] = agent.select_action(obs, mask, training=training)
        return actions

    def store_transitions(
        self,
        observations: Dict[int, np.ndarray],
        actions: Dict[int, int],
        team_reward: float,
        next_observations: Dict[int, np.ndarray],
        next_action_masks: Dict[int, np.ndarray],
        done: bool
    ):
        """Stores team reward transition for all robot agents."""
        for r_id, agent in self.agents.items():
            agent.store_transition(
                observations[r_id],
                actions[r_id],
                team_reward,
                next_observations[r_id],
                next_action_masks[r_id],
                done
            )

    def train_step(self) -> Dict[int, Optional[float]]:
        """Trains all agents for one step."""
        losses = {}
        for r_id, agent in self.agents.items():
            losses[r_id] = agent.train_step()
        return losses

    # Single-agent compatibility wrappers
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, training: bool = True) -> int:
        return self.agents[0].select_action(observation, action_mask, training=training)

    def store_transition(self, obs, action, reward, next_obs, next_mask, done):
        self.agents[0].store_transition(obs, action, reward, next_obs, next_mask, done)

    def save(self, directory: str):
        """Saves all agent weights to directory."""
        os.makedirs(directory, exist_ok=True)
        for r_id, agent in self.agents.items():
            agent.save(os.path.join(directory, f"agent_{r_id}.pt"))

    def load(self, directory: str):
        """Loads all agent weights from directory."""
        for r_id, agent in self.agents.items():
            model_path = os.path.join(directory, f"agent_{r_id}.pt")
            if os.path.exists(model_path):
                agent.load(model_path)
