import hashlib
import json
import os
import unittest

import numpy as np

from config import DEFAULT_CONFIG as cfg
from run_experiments import evaluate_marl_on_env
from src.communication.channel import CommunicationChannel
from src.env.warehouse_env import WarehouseMRTAEnv
from src.marl.shared_dqfd import SharedDQNController, augment

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "shared_dqfd")


class SharedMARLTests(unittest.TestCase):
    def test_checkpoint_matches_metadata_hash(self):
        with open(os.path.join(MODEL_DIR, "training_metadata.json")) as f:
            meta = json.load(f)
        with open(os.path.join(MODEL_DIR, "shared_dqfd.pt"), "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), meta["checkpoint_sha256"])

    def test_actions_always_respect_the_mask(self):
        ctrl = SharedDQNController(cfg); ctrl.load(MODEL_DIR)
        env = WarehouseMRTAEnv(CommunicationChannel("L2_Medium", cfg), cfg)
        obs, info = env.reset(seed=3); masks = info["action_masks"]; done = False
        while not done:
            acts = ctrl.select_actions(obs, masks)
            for r, a in acts.items():
                self.assertTrue(masks[r][a])
            obs, _, done, si = env.step(acts); masks = si["action_masks"]

    def test_robot_id_is_appended_to_the_observation(self):
        o = augment(np.zeros(87, dtype=np.float32), 1, 3)
        self.assertEqual(o.shape[0], 90)
        self.assertEqual(o[87:].tolist(), [0.0, 1.0, 0.0])

    def test_aligned_reward_pays_only_for_on_time_completion(self):
        env = WarehouseMRTAEnv(CommunicationChannel("L4_Full", cfg), cfg)
        self.assertEqual(env.reward_mode, "original")          # original IDQN results stay reproducible
        env.reward_mode = "aligned"
        obs, info = env.reset(seed=1)
        wait = {r: cfg.n_tasks for r in obs}
        _, reward, _, _ = env.step(wait)
        self.assertEqual(reward, 0.0)                          # nothing moved, nothing sent

    def test_evaluation_runs_and_is_deterministic(self):
        ctrl = SharedDQNController(cfg); ctrl.load(MODEL_DIR)
        a = evaluate_marl_on_env(ctrl, "L2_Medium", 2, cfg)["total_cost"]
        b = evaluate_marl_on_env(ctrl, "L2_Medium", 2, cfg)["total_cost"]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
