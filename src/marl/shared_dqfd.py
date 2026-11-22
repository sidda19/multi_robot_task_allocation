"""Cooperative MARL with parameter sharing, trained from demonstrations + team-reward Q-learning (DQfD-style).

Why this exists
---------------
The original independent-DQN checkpoint (`marl_agent.py`) learned nothing beyond what the action mask already
gives (it equals a random masked policy). Plain DQN with a shared network also failed in our tests (it collapsed
to waiting). This module keeps the proposal's MARL structure and changes only HOW the policy is trained:

* observation  : the proposal's 87-d local observation (+ a 3-d one-hot robot id for the shared network)
* actions      : task j or WAIT, infeasible actions masked
* reward       : shared team reward of the form of proposal Eq. 12/18,
                 r = 50 * (on-time completions) - travel - 10 * conflicts - 0.1 * messages
                 (50 = P_u, 10 = P_c: the same terms as the evaluation cost; env.reward_mode = 'aligned')
* communication: fixed externally per episode (levels are cycled during training); nothing is learned about messages
* execution    : decentralised; every robot uses only its own observation and mask
* training     : one Q-network shared by the three robots (Double DQN, 1-step TD on the team reward) plus a
                 large-margin supervised loss on demonstrations of a hand-written online greedy rule that needs
                 no messages (`OnlineGreedyPolicy(tie_break="convention")`). Demonstrations are used only to
                 train; after that the agents act on their own and keep learning from the team reward.
"""
import os
import random
import time
import json
import hashlib
from collections import deque
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import DEFAULT_CONFIG, SimulationConfig
from src.communication.channel import CommunicationChannel
from src.env.warehouse_env import WarehouseMRTAEnv
from src.marl.q_network import MaskedQNetwork

LEVELS = ["L0_None", "L1_Low", "L2_Medium", "L3_High", "L4_Full"]


def augment(obs: np.ndarray, robot_id: int, n_robots: int) -> np.ndarray:
    one_hot = np.zeros(n_robots, dtype=np.float32)
    one_hot[robot_id] = 1.0
    return np.concatenate([obs, one_hot])


class SharedDQNController:
    """One Q-network shared by all robots; each robot acts from its own observation and mask."""

    def __init__(self, config: SimulationConfig = DEFAULT_CONFIG, hidden: int = 128, lr: float = 5e-4,
                 gamma: float = 0.97, target_every: int = 200):
        self.config = config
        self.n_robots = config.n_robots
        self.obs_dim = 7 + config.n_tasks * 8 + config.n_robots
        self.action_dim = config.n_tasks + 1
        self.hidden = hidden
        self.q = MaskedQNetwork(self.obs_dim, self.action_dim, hidden)
        self.target = MaskedQNetwork(self.obs_dim, self.action_dim, hidden)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.target_every, self.steps, self.eps = gamma, target_every, 0, 0.0

    def bind(self, env) -> None:      # same interface as the reference policies
        pass

    def select_actions(self, obs: Dict[int, np.ndarray], masks: Dict[int, np.ndarray], training: bool = False) -> Dict[int, int]:
        actions = {}
        for r, o in obs.items():
            mask = np.asarray(masks[r], dtype=bool)
            if training and random.random() < self.eps:
                actions[r] = int(random.choice(np.flatnonzero(mask)))
                continue
            with torch.no_grad():
                q = self.q(torch.tensor(augment(o, r, self.n_robots)).unsqueeze(0), torch.tensor(mask).unsqueeze(0))
            actions[r] = int(q.argmax(1).item())
        return actions

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save({"q": self.q.state_dict(), "hidden": self.hidden}, os.path.join(directory, "shared_dqfd.pt"))

    def load(self, directory: str) -> None:
        ckpt = torch.load(os.path.join(directory, "shared_dqfd.pt"), map_location="cpu")
        self.q.load_state_dict(ckpt["q"])
        self.target.load_state_dict(ckpt["q"])


def _transition(ctrl, obs, acts, rew, nobs, masks, nmasks, done):
    n = ctrl.n_robots
    return [(augment(obs[r], r, n), acts[r], rew, augment(nobs[r], r, n),
             np.asarray(masks[r], bool), np.asarray(nmasks[r], bool), float(done)) for r in obs]


def _learn(ctrl, demo, own, batch=128, margin=10.0, lam=1.0):
    nd = min(len(demo), batch // 2) if len(own) >= batch // 2 else min(len(demo), batch)
    ns = min(len(own), batch - nd)
    b = random.sample(demo, nd) + (random.sample(own, ns) if ns else [])
    is_demo = torch.tensor([1.0] * nd + [0.0] * ns)
    s = torch.tensor(np.stack([x[0] for x in b])); a = torch.tensor([x[1] for x in b])
    r = torch.tensor([x[2] for x in b], dtype=torch.float32)
    s2 = torch.tensor(np.stack([x[3] for x in b]))
    m = torch.tensor(np.stack([x[4] for x in b])); m2 = torch.tensor(np.stack([x[5] for x in b]))
    d = torch.tensor([x[6] for x in b], dtype=torch.float32)
    qa = ctrl.q(s)
    q = qa.gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():                                   # Double DQN target
        a2 = ctrl.q(s2, m2).argmax(1, keepdim=True)
        y = r + (1 - d) * ctrl.gamma * ctrl.target(s2, m2).gather(1, a2).squeeze(1)
    td = nn.functional.smooth_l1_loss(q, y)
    marg = torch.full_like(qa, margin); marg.scatter_(1, a.unsqueeze(1), 0.0)
    qm = torch.where(m, qa + marg, torch.full_like(qa, -1e9))
    sup = ((qm.max(1).values - q) * is_demo).sum() / is_demo.sum().clamp(min=1)
    loss = td + lam * sup
    ctrl.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ctrl.q.parameters(), 10); ctrl.opt.step()
    ctrl.steps += 1
    if ctrl.steps % ctrl.target_every == 0:
        ctrl.target.load_state_dict(ctrl.q.state_dict())


def train_shared_dqfd(demo_episodes: int = 400, pretrain_steps: int = 12000, online_episodes: int = 1500,
                      seed: int = 2026, save_dir: str = "models/shared_dqfd", config: SimulationConfig = DEFAULT_CONFIG,
                      verbose: bool = True) -> Dict[str, object]:
    """Deterministic training. Training scenarios use seeds >= 300000 (never the evaluation seeds 1-10)."""
    from src.baselines.online_policies import OnlineGreedyPolicy   # local import: avoids a cycle
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    t0 = time.perf_counter()
    ctrl = SharedDQNController(config)
    envs = {l: WarehouseMRTAEnv(CommunicationChannel(l, config), config) for l in LEVELS}
    for e in envs.values():
        e.reward_mode = "aligned"
    teacher = OnlineGreedyPolicy(config, tie_break="convention")
    demo, own = [], []
    for ep in range(demo_episodes):                                   # 1) demonstrations
        env = envs[LEVELS[ep % len(LEVELS)]]
        obs, info = env.reset(seed=300000 + ep); masks = info["action_masks"]; teacher.bind(env); done = False
        while not done:
            acts = teacher.select_actions(obs, masks)
            nobs, rew, done, si = env.step(acts)
            demo += _transition(ctrl, obs, acts, rew, nobs, masks, si["action_masks"], done)
            obs, masks = nobs, si["action_masks"]
    for _ in range(pretrain_steps):                                   # 2) pre-training on demonstrations
        _learn(ctrl, demo, own)
    ctrl.target.load_state_dict(ctrl.q.state_dict())
    returns = []
    for ep in range(online_episodes):                                 # 3) own experience, team reward
        ctrl.eps = 0.05
        env = envs[LEVELS[ep % len(LEVELS)]]
        obs, info = env.reset(seed=400000 + ep); masks = info["action_masks"]; done = False; total = 0.0
        while not done:
            acts = ctrl.select_actions(obs, masks, training=True)
            nobs, rew, done, si = env.step(acts)
            own += _transition(ctrl, obs, acts, rew, nobs, masks, si["action_masks"], done)
            _learn(ctrl, demo, own)
            obs, masks = nobs, si["action_masks"]; total += rew
        returns.append(total)
        if verbose and (ep + 1) % 250 == 0:
            print(f"online episode {ep + 1}/{online_episodes} | mean team return (last 250): {np.mean(returns[-250:]):.1f}", flush=True)
    ctrl.save(save_dir)
    with open(os.path.join(save_dir, "shared_dqfd.pt"), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    meta = {"method": "shared-parameter Double DQN + demonstrations (DQfD-style)", "seed": seed,
            "demo_episodes": demo_episodes, "pretrain_steps": pretrain_steps, "online_episodes": online_episodes,
            "hidden": ctrl.hidden, "gamma": ctrl.gamma, "lr": 5e-4, "batch": 128, "margin": 10.0,
            "online_epsilon": 0.05, "reward": "50*on-time completions - travel - 10*conflicts - 0.1*messages",
            "teacher": "OnlineGreedyPolicy(tie_break='convention'), training only",
            "training_scenario_seeds": ">=300000 (evaluation seeds are 1-10)",
            "training_time_s": round(time.perf_counter() - t0, 1), "checkpoint_sha256": digest}
    with open(os.path.join(save_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return {"controller": ctrl, "metadata": meta}
