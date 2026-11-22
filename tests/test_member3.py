import os
import sys
import unittest
import numpy as np

# Ensure libs in sys.path
_libs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "libs"))
if os.path.exists(_libs_path) and _libs_path not in sys.path:
    sys.path.insert(0, _libs_path)

import torch
from src.env.entities import Location, Task, Robot
from src.env.warehouse_env import WarehouseMRTAEnv
from src.communication.channel import CommunicationChannel
from src.marl.q_network import MaskedQNetwork
from src.marl.marl_agent import IDQNAgent, CooperativeMARLController, ReplayBuffer
from config import DEFAULT_CONFIG, SimulationConfig
from run_experiments import _missed_deadline


class TestMember3Components(unittest.TestCase):
    def setUp(self):
        self.config = DEFAULT_CONFIG
        self.channel = CommunicationChannel("L2_Medium", self.config)
        self.env = WarehouseMRTAEnv(self.channel, self.config)

    # 1. Environment reset and reproducibility
    def test_01_environment_reset_and_reproducibility(self):
        obs1, info1 = self.env.reset(seed=123)
        obs2, info2 = self.env.reset(seed=123)

        for r_id in obs1:
            np.testing.assert_array_almost_equal(obs1[r_id], obs2[r_id])
            np.testing.assert_array_equal(info1["action_masks"][r_id], info2["action_masks"][r_id])

        # Different seed produces different initial setup
        obs3, _ = self.env.reset(seed=999)
        self.assertFalse(np.allclose(obs1[0], obs3[0]))

    # 2. Correct 87-dimensional observation space
    def test_02_environment_observation_dimensions(self):
        obs, info = self.env.reset(seed=42)
        expected_dim = 7 + self.config.n_tasks * 8  # 7 + 80 = 87

        self.assertEqual(len(obs), self.config.n_robots)
        for r_id, vector in obs.items():
            self.assertEqual(vector.shape, (expected_dim,))
            self.assertTrue(np.all(vector >= 0.0) and np.all(vector <= 1.0))

    # 3. Action masking structure
    def test_03_action_masking_structure(self):
        _, info = self.env.reset(seed=42)
        masks = info["action_masks"]
        M = self.config.n_tasks

        for r_id, mask in masks.items():
            self.assertEqual(len(mask), M + 1)
            # WAIT action (index M) must ALWAYS be valid
            self.assertTrue(mask[M])

    # 4. Capability constraints in action masking
    def test_04_capability_constraints(self):
        self.env.reset(seed=42)
        masks = self.env._get_action_masks()

        for j, task in enumerate(self.env.tasks):
            if task.capability_req == "B":
                self.assertFalse(masks[0][j])  # Robot 0 has {"A"} only
            elif task.capability_req == "A":
                self.assertFalse(masks[1][j])  # Robot 1 has {"B"} only

    # 5. Deadline constraints in action masking
    def test_05_deadline_constraints(self):
        self.env.reset(seed=42)
        self.env.tasks[0].deadline = 0  # Impossible deadline
        masks = self.env._get_action_masks()

        for r_id in range(self.config.n_robots):
            self.assertFalse(masks[r_id][0])

    # 6. Robot capacity constraints
    def test_06_robot_capacity_constraints(self):
        self.env.reset(seed=42)
        robot = self.env.robots[0]
        robot.tasks_assigned = [991, 992, 993]  # Capacity exhausted
        masks = self.env._get_action_masks()

        M = self.config.n_tasks
        self.assertTrue(masks[0][M])
        for j in range(M):
            self.assertFalse(masks[0][j])

    # 7. No global state leakage: two robots can have different local beliefs about the same task
    def test_07_no_global_leakage_and_distinct_local_beliefs(self):
        ch = CommunicationChannel("L0_None", self.config)
        env = WarehouseMRTAEnv(ch, self.config)
        env.reset(seed=42)

        # In L0, robot 0 claims task 0 physically in ground truth
        env.tasks[0].assigned_robot = 0
        masks = env._get_action_masks()

        # Because robot 2 has dual capabilities {A, B} and has NOT received any message from robot 0,
        # robot 2's action mask must NOT be blinded by global assignment:
        # Task 0 must remain valid in robot 2's mask!
        if env.tasks[0].deadline > 10:
            self.assertTrue(masks[2][0])

    # 8. Communication delivery updates local beliefs at L2
    def test_08_communication_updates_local_beliefs(self):
        ch = CommunicationChannel("L2_Medium", self.config)
        env = WarehouseMRTAEnv(ch, self.config)
        env.reset(seed=42)

        # Robot 0 claims task 0 at step 1 (step 1 is not a multiple of k=5)
        ch.broadcast(sender_id=0, msg_type="CLAIM", payload={"task_id": 0}, current_step=1)
        
        # At step 1 (before flush at step 5), robot 2 does NOT know task 0 is claimed
        env.current_step = 1
        env._process_incoming_messages()
        self.assertNotIn(0, env.local_known_claimed[2])

        # At step 5 (flush step), robot 2 receives the message and updates its belief
        env.current_step = 5
        ch.flush_buffer_if_period(5)
        env._process_incoming_messages()
        self.assertIn(0, env.local_known_claimed[2])

        # Now robot 2's action mask excludes task 0
        masks = env._get_action_masks()
        self.assertFalse(masks[2][0])

    # 9. Realistic multi-step task execution
    def test_09_multistep_physical_execution(self):
        env = WarehouseMRTAEnv(CommunicationChannel("L4_Full", self.config), self.config)
        env.reset(seed=42)

        robot = env.robots[0]
        initial_battery = robot.battery
        initial_pos = Location(robot.position.x, robot.position.y)

        # Select a feasible task for robot 0
        task_id = next(t.id for t in env.tasks if t.capability_req == "A")
        task = env.tasks[task_id]

        # Step 1: select task
        obs, reward, done, info = env.step({0: task_id, 1: 10, 2: 10})
        self.assertFalse(task.completed)

        # Step multiple times: robot moves towards pickup
        for _ in range(5):
            env.step({0: 10, 1: 10, 2: 10})

        self.assertGreater(robot.total_travel_distance, 0)
        self.assertLess(robot.battery, initial_battery)

    # 10. Shared team reward calculation
    def test_10_team_reward_calculation(self):
        env = WarehouseMRTAEnv(CommunicationChannel("L4_Full", self.config), self.config)
        env.reset(seed=42)

        # 3 WAIT actions: no travel, no completion, no conflicts, no messages
        _, reward, _, info = env.step({0: 10, 1: 10, 2: 10})
        self.assertEqual(reward, 0.0)

    # 11. Q-Network forward pass with 87 dimensions
    def test_11_q_network_forward_pass(self):
        obs_dim = 87
        action_dim = 11
        net = MaskedQNetwork(obs_dim=obs_dim, action_dim=action_dim)
        dummy_obs = torch.rand(4, obs_dim)
        dummy_mask = torch.ones(4, action_dim, dtype=torch.bool)
        dummy_mask[:, 0] = False  # Mask action 0

        q_vals = net(dummy_obs, dummy_mask)
        self.assertEqual(q_vals.shape, (4, action_dim))
        self.assertTrue(torch.all(q_vals[:, 0] < -1e8))

    # 12. Replay buffer operations
    def test_12_replay_buffer(self):
        buf = ReplayBuffer(capacity=100)
        obs = np.zeros(87, dtype=np.float32)
        mask = np.ones(11, dtype=bool)
        buf.push(obs, 1, 10.0, obs, mask, False)

        self.assertEqual(len(buf), 1)
        o, a, r, no, nm, d = buf.sample(1)
        self.assertEqual(o.shape, (1, 87))
        self.assertEqual(a.item(), 1)
        self.assertEqual(r.item(), 10.0)

    # 13. IDQN Agent training update
    def test_13_training_update(self):
        agent = IDQNAgent(agent_id=0, obs_dim=87, action_dim=11, batch_size=4)
        obs = np.random.rand(87).astype(np.float32)
        mask = np.ones(11, dtype=bool)

        for _ in range(10):
            agent.replay_buffer.push(obs, 0, 1.0, obs, mask, False)

        loss = agent.train_step()
        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss, 0.0)

    # 14. Common physical execution sequential constraints
    def test_14_common_physical_execution_sequential(self):
        env = WarehouseMRTAEnv(CommunicationChannel("L4_Full", self.config), self.config)
        env.reset(seed=42)

        # Robot 0 is assigned 3 tasks
        assignments = {0: [0, 2, 4], 1: [], 2: []}
        res = env.run_physical_execution(assignments)

        # Tasks must be executed sequentially, requiring positive travel distance
        self.assertIn("physically_completed_count", res)
        self.assertIn("physical_completion_rate", res)
        self.assertGreater(res["total_travel_distance"], 0)

    # 15. Uncommunicated claims produce physical conflicts at L0
    def test_15_uncommunicated_claims_produce_physical_conflicts(self):
        ch = CommunicationChannel("L0_None", self.config)
        env = WarehouseMRTAEnv(ch, self.config)
        env.reset(seed=42)

        # Robot 0 and Robot 2 both claim task 0 (requiring A, both capable) simultaneously
        obs, reward, done, info = env.step({0: 0, 1: 10, 2: 0})
        self.assertGreater(info["conflicts_this_step"], 0)
        # Winner (Robot 0 or 2 with lower cost) owns the task
        self.assertIsNotNone(env.tasks[0].assigned_robot)


    # 16. Static physical replay does not add unreported communication messages
    def test_16_static_replay_does_not_emit_claim_messages(self):
        channel = CommunicationChannel("L4_Full", self.config)
        env = WarehouseMRTAEnv(channel, self.config)
        env.reset(seed=7)
        env.run_physical_execution({0: [0], 1: [], 2: []})
        self.assertEqual(channel.total_messages, 0)

    # 17. Physical movement stops when the robot has no battery
    def test_17_battery_depletion_stops_physical_movement(self):
        env = WarehouseMRTAEnv(CommunicationChannel("L4_Full", self.config), self.config)
        env.reset(seed=42)
        robot = env.robots[0]
        task = next(t for t in env.tasks if t.capability_req == "A")
        robot.battery = 0.0
        initial_position = (robot.position.x, robot.position.y)
        env.step({0: task.id, 1: self.config.n_tasks, 2: self.config.n_tasks})
        self.assertEqual((robot.position.x, robot.position.y), initial_position)
        self.assertFalse(task.completed)

    # 18. A completed-after-deadline task is still a missed deadline
    def test_18_late_completion_counts_as_missed_deadline(self):
        task = Task(
            id=99, pickup=Location(0, 0), delivery=Location(1, 1),
            capability_req="A", deadline=5, assigned_robot=0,
            completed=True, completion_time=6
        )
        self.assertTrue(_missed_deadline(task))

    # 19. Static replay rejects duplicate ownership of one task
    def test_19_static_replay_rejects_duplicate_task_assignments(self):
        env = WarehouseMRTAEnv(CommunicationChannel("L4_Full", self.config), self.config)
        env.reset(seed=42)
        with self.assertRaises(ValueError):
            env.run_physical_execution({0: [0], 1: [0], 2: []})

if __name__ == "__main__":
    unittest.main()
