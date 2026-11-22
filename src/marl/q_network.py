import sys
import os

# Automatically include local libs if present
_libs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "libs"))
if os.path.exists(_libs_path) and _libs_path not in sys.path:
    sys.path.insert(0, _libs_path)

import torch
import torch.nn as nn
from typing import Optional

class MaskedQNetwork(nn.Module):
    """
    Lightweight 2-layer MLP Deep Q-Network with Action Masking.
    Architecture:
      Input (obs_dim, default 87) -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(action_dim, default 11)
    Masks infeasible task choices (deadlines missed, capacity full, capability mismatch, busy)
    by setting their Q-values to -1e9 so that argmax will never select them.
    """
    def __init__(self, obs_dim: int = 87, action_dim: int = 11, hidden_dim: int = 64):
        super(MaskedQNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional boolean action mask.
        mask: Tensor of shape matching (..., action_dim), where True indicates valid actions.
        """
        q_vals = self.net(obs)
        if mask is not None:
            large_neg = torch.tensor(-1e9, dtype=q_vals.dtype, device=q_vals.device)
            q_vals = torch.where(mask, q_vals, large_neg)
        return q_vals
