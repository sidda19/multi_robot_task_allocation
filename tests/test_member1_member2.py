import unittest
import copy
import numpy as np
from src.env.entities import Location, Task, Robot
from src.baselines.exact_ilp import ExactILPSolver
from src.baselines.centralized_greedy import CentralizedGreedyAllocator
from src.baselines.distributed_greedy import DistributedGreedyAllocator
from src.baselines.auction import AuctionAllocator
from src.communication.channel import CommunicationChannel
from src.evaluation.metrics import compute_metrics
from config import DEFAULT_CONFIG

class TestMember1AndMember2Baselines(unittest.TestCase):
    def setUp(self):
        # 3 robots (Member 1 spec)
        self.robots = [
            Robot(id=0, position=Location(1, 1), capabilities={"A"}, capacity_limit=3),
            Robot(id=1, position=Location(2, 2), capabilities={"B"}, capacity_limit=3),
            Robot(id=2, position=Location(5, 5), capabilities={"A", "B"}, capacity_limit=3)
        ]
        
        # 4 diverse tasks with deadlines
        self.tasks = [
            Task(id=0, pickup=Location(1, 2), delivery=Location(3, 3), capability_req="A", deadline=30),
            Task(id=1, pickup=Location(2, 1), delivery=Location(4, 2), capability_req="B", deadline=30),
            Task(id=2, pickup=Location(6, 6), delivery=Location(8, 8), capability_req="A", deadline=30),
            Task(id=3, pickup=Location(5, 4), delivery=Location(7, 5), capability_req="B", deadline=30)
        ]

    # --- MEMBER 1 TESTS ---
    def test_member1_exact_ilp_solver(self):
        """Verify Exact ILP finds optimal solution and respects constraints."""
        solver = ExactILPSolver(DEFAULT_CONFIG)
        res = solver.solve(self.robots, self.tasks)
        
        self.assertTrue(res["success"])
        self.assertGreater(res["optimal_cost"], 0.0)
        self.assertEqual(res["completion_rate"], 1.0)
        
        # Verify robot capacity
        for r_id, assigned_tasks in res["assignments"].items():
            self.assertLessEqual(len(assigned_tasks), self.robots[r_id].capacity_limit)
            
        # Verify capability matching
        for r_id, assigned_tasks in res["assignments"].items():
            for t_id in assigned_tasks:
                task = next(t for t in self.tasks if t.id == t_id)
                self.assertIn(task.capability_req, self.robots[r_id].capabilities)

    def test_member1_exact_ilp_cost_decomposition(self):
        """Verify ILP cost decomposition: total_cost = assignment_cost + unassigned_penalty + conflict_penalty."""
        solver = ExactILPSolver(DEFAULT_CONFIG)
        res = solver.solve(self.robots, self.tasks)
        
        expected_total = res["assignment_cost"] + res["unassigned_penalty"] + res["conflict_penalty"]
        self.assertAlmostEqual(res["total_cost"], expected_total, places=4)
        self.assertEqual(res["conflict_penalty"], 0.0)

    def test_member1_exact_ilp_infeasible_deadline(self):
        """Verify that a task with an impossible deadline is marked unassigned."""
        tight_tasks = [
            Task(id=0, pickup=Location(9, 9), delivery=Location(0, 0), capability_req="A", deadline=1)
        ]
        solver = ExactILPSolver(DEFAULT_CONFIG)
        res = solver.solve(self.robots, tight_tasks)
        self.assertTrue(res["success"])
        self.assertIn(0, res["unassigned_tasks"])
        self.assertEqual(res["unassigned_penalty"], DEFAULT_CONFIG.penalty_unassigned)

    def test_member1_centralized_greedy_taskwise(self):
        """Verify Centralized Greedy executes task-wise selection according to Eq. 14-15."""
        cg = CentralizedGreedyAllocator(DEFAULT_CONFIG)
        res = cg.solve(self.robots, self.tasks)
        
        self.assertEqual(res["assignment_rate"], 1.0)
        self.assertGreater(res["total_cost"], 0.0)
        for r_id, assigned_tasks in res["assignments"].items():
            self.assertLessEqual(len(assigned_tasks), self.robots[r_id].capacity_limit)
            for t_id in assigned_tasks:
                task = next(t for t in self.tasks if t.id == t_id)
                self.assertIn(task.capability_req, self.robots[r_id].capabilities)

    # --- MEMBER 2 TESTS & PHASE 1 REQUIREMENTS ---
    def test_member2_communication_channel(self):
        """Verify CommunicationChannel obeys L0-L4 intervals and budgets."""
        # L0: No communication allowed
        ch_l0 = CommunicationChannel("L0_None", DEFAULT_CONFIG)
        ok = ch_l0.broadcast(sender_id=0, msg_type="CLAIM", payload={"task_id": 0}, current_step=0)
        self.assertFalse(ok)
        self.assertEqual(ch_l0.total_messages, 0)
        
        # L2: Periodic sync every 5 steps
        ch_l2 = CommunicationChannel("L2_Medium", DEFAULT_CONFIG)
        ok = ch_l2.broadcast(sender_id=0, msg_type="CLAIM", payload={"task_id": 0}, current_step=1)
        self.assertTrue(ok)
        self.assertEqual(len(ch_l2.outbox_buffer), 1)
        
        # At step 5, buffer flushes
        ch_l2.flush_buffer_if_period(5)
        inbox_1 = ch_l2.receive_messages(1)
        self.assertEqual(len(inbox_1), 1)
        self.assertEqual(inbox_1[0].payload["task_id"], 0)

    def test_member2_distributed_greedy_no_teammate_info_at_l0(self):
        """Verify that at L0, robots receive zero teammate claim messages."""
        ch_l0 = CommunicationChannel("L0_None", DEFAULT_CONFIG)
        dg_l0 = DistributedGreedyAllocator(ch_l0, DEFAULT_CONFIG)
        res_l0 = dg_l0.solve(self.robots, self.tasks)
        
        self.assertEqual(res_l0["messages_sent"], 0)
        self.assertEqual(ch_l0.total_messages, 0)

    def test_member2_distributed_greedy_lower_cost_winner_and_tie_breaking(self):
        """Verify that conflict resolution prioritizes lower estimated cost, using ID as tie-break."""
        # Task 0 at (1, 2) is equidistant or closer to robot 0 at (1, 1) than robot 2 at (5, 5)
        ch_l4 = CommunicationChannel("L4_Full", DEFAULT_CONFIG)
        dg_l4 = DistributedGreedyAllocator(ch_l4, DEFAULT_CONFIG)
        res_l4 = dg_l4.solve(self.robots, self.tasks)
        
        # Task 0 should be assigned to robot 0 (cost is lower)
        self.assertIn(0, res_l4["assignments"][0])

    def test_member2_auction_baseline(self):
        """Verify Sequential Single-Item Auction under L0 and L4."""
        # L0 Auction: no bid communication -> conflict penalty
        ch_l0 = CommunicationChannel("L0_None", DEFAULT_CONFIG)
        auc_l0 = AuctionAllocator(ch_l0, DEFAULT_CONFIG)
        res_l0 = auc_l0.solve(self.robots, self.tasks)
        self.assertGreater(res_l0["conflict_count"], 0)
        
        # L4 Auction: bids exchanged -> 0 conflicts
        ch_l4 = CommunicationChannel("L4_Full", DEFAULT_CONFIG)
        auc_l4 = AuctionAllocator(ch_l4, DEFAULT_CONFIG)
        res_l4 = auc_l4.solve(self.robots, self.tasks)
        self.assertEqual(res_l4["conflict_count"], 0)

    def test_coordinator_message_budget_is_enforced(self):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg.COMM_LEVELS["L4_Full"]["budget"] = 1
        ch = CommunicationChannel("L4_Full", cfg)
        self.assertTrue(ch.send_to_coordinator(0, "BID", {"task_id": 0, "bid": 1.0}, 0))
        self.assertFalse(ch.send_to_coordinator(0, "BID", {"task_id": 1, "bid": 2.0}, 0))
        received = ch.receive_coordinator_messages()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload["task_id"], 0)
        self.assertEqual(ch.total_messages, 1)

    def test_auction_uses_only_budgeted_delivered_bids(self):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg.COMM_LEVELS["L4_Full"]["budget"] = 1
        channel = CommunicationChannel("L4_Full", cfg)
        result = AuctionAllocator(channel, cfg).solve(self.robots, self.tasks)
        self.assertLessEqual(result["messages_sent"], cfg.n_robots)
        self.assertLess(len([t for values in result["assignments"].values() for t in values]), len(self.tasks))
        self.assertEqual(result["assignment_objective_cost"],
                         result["assignment_cost"] + result["unassigned_penalty"])

    def test_distributed_greedy_three_robot_conflict_is_counted_once_per_loser(self):
        robots = [
            Robot(id=i, position=Location(0, 0), capabilities={"A"}, capacity_limit=1)
            for i in range(3)
        ]
        tasks = [
            Task(id=0, pickup=Location(1, 0), delivery=Location(2, 0),
                 capability_req="A", deadline=30)
        ]
        channel = CommunicationChannel("L0_None", DEFAULT_CONFIG)
        result = DistributedGreedyAllocator(channel, DEFAULT_CONFIG).solve(robots, tasks)
        self.assertEqual(result["conflict_count"], 2)
        self.assertEqual(sum(len(v) for v in result["assignments"].values()), 1)

    def test_static_assignment_gap_excludes_conflict_penalty(self):
        result = {
            "total_cost": 50.0,
            "assignment_objective_cost": 30.0,
            "assignment_cost": 20.0,
            "assignment_rate": 1.0,
            "conflict_count": 2,
            "runtime_ms": 1.0
        }
        metrics = compute_metrics("Distributed Greedy", "L0_None", result, 25.0)
        self.assertAlmostEqual(metrics.optimality_gap_percent, 20.0)

    def test_metrics_evaluation(self):
        """Verify compute_metrics calculates optimality gap correctly."""
        dummy_res = {
            "total_cost": 30.0,
            "assignment_cost": 30.0,
            "assignment_objective_cost": 30.0,
            "completion_rate": 1.0,
            "assignment_rate": 1.0,
            "conflict_count": 0,
            "messages_sent": 5,
            "runtime_ms": 1.5
        }
        metrics = compute_metrics("Auction", "L3_High", dummy_res, optimal_cost=25.0)
        self.assertEqual(metrics.optimality_gap_percent, 20.0)
        online_metrics = compute_metrics("MARL", "L2_Medium", {"total_cost": 20.0}, optimal_cost=25.0)
        self.assertIsNone(online_metrics.optimality_gap_percent)
        self.assertEqual(online_metrics.to_dict()["optimality_gap_%"], "N/A")

        negative_result = {
            "assignment_objective_cost": 20.0,
            "assignment_rate": 1.0,
            "runtime_ms": 1.0
        }
        negative_metrics = compute_metrics(
            "Static Solver", "L2_Medium", negative_result, optimal_cost=25.0
        )
        self.assertAlmostEqual(negative_metrics.optimality_gap_percent, -20.0)

if __name__ == "__main__":
    unittest.main()
