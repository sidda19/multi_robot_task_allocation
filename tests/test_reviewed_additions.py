"""Tests for the review additions: sequence-aware exact reference, reference
online policies, the allocation-clock fix, and the accounting identity that the
review found violated in one of the earlier packages."""
import copy
import unittest

import numpy as np

from config import DEFAULT_CONFIG
from run_experiments import (
    evaluate_online_policy_on_env,
    execute_static_allocation_physically,
    generate_scenario,
)
from src.baselines.auction import AuctionAllocator
from src.baselines.centralized_greedy import CentralizedGreedyAllocator
from src.baselines.exact_ilp import ExactILPSolver
from src.baselines.exact_ilp_sequenced import SequenceAwareExactSolver
from src.baselines.online_policies import OnlineGreedyPolicy, RandomMaskedPolicy
from src.communication.channel import CommunicationChannel
from src.env.entities import Location, Robot, Task

CFG = DEFAULT_CONFIG


def _replay(assignments, seed):
    return execute_static_allocation_physically(
        assignments, CommunicationChannel("L4_Full", CFG), seed, CFG
    )


class TestSequenceAwareReference(unittest.TestCase):
    def test_plan_replays_exactly_with_no_missed_deadline(self):
        """The reference must be exact for the simulator: predicted cost == replayed cost."""
        for seed in range(1, 41):
            robots, tasks = generate_scenario(seed)
            res = SequenceAwareExactSolver(CFG).solve(robots, tasks)
            phys = _replay(res["assignments"], seed)
            self.assertEqual(phys["missed_deadlines_count"], 0, f"seed {seed}")
            self.assertAlmostEqual(phys["total_physical_cost"], res["optimal_cost"], places=6)

    def test_no_method_beats_the_reference(self):
        """Greedy plans, the assignment-ILP plan and online policies can never be cheaper."""
        for seed in range(1, 11):
            robots, tasks = generate_scenario(seed)
            ref = SequenceAwareExactSolver(CFG).solve(copy.deepcopy(robots), copy.deepcopy(tasks))["optimal_cost"]
            ilp_plan = ExactILPSolver(CFG).solve(copy.deepcopy(robots), copy.deepcopy(tasks))["assignments"]
            cg_plan = CentralizedGreedyAllocator(CFG).solve(copy.deepcopy(robots), copy.deepcopy(tasks))["assignments"]
            self.assertGreaterEqual(_replay(ilp_plan, seed)["total_physical_cost"], ref - 1e-9)
            self.assertGreaterEqual(_replay(cg_plan, seed)["total_physical_cost"], ref - 1e-9)
            for policy in (OnlineGreedyPolicy(CFG), RandomMaskedPolicy(seed, CFG)):
                cost = evaluate_online_policy_on_env(policy, "L4_Full", seed, CFG)["total_cost"]
                self.assertGreaterEqual(cost, ref - 1e-9)

    def test_capacity_capability_and_unique_ownership(self):
        robots, tasks = generate_scenario(3)
        res = SequenceAwareExactSolver(CFG).solve(robots, tasks)
        seen = []
        for rid, seq in res["assignments"].items():
            self.assertLessEqual(len(seq), robots[rid].capacity_limit)
            for tid in seq:
                self.assertIn(tasks[tid].capability_req, robots[rid].capabilities)
            seen += seq
        self.assertEqual(len(seen), len(set(seen)))

    def test_sequence_matters_where_assignment_ilp_is_optimistic(self):
        """Two tasks with the same deadline that one robot can only serve one after the other."""
        robot = Robot(id=0, position=Location(0, 0), capabilities={"A"}, capacity_limit=2)
        t0 = Task(id=0, pickup=Location(0, 0), delivery=Location(5, 0), capability_req="A", duration=2, deadline=9)
        t1 = Task(id=1, pickup=Location(0, 1), delivery=Location(5, 1), capability_req="A", duration=2, deadline=9)
        # Assignment ILP: each task alone is feasible from t = 0 -> assigns both.
        ilp = ExactILPSolver(CFG).solve([robot], [t0, t1])
        self.assertEqual(len(ilp["unassigned_tasks"]), 0)
        # Sequence-aware: the second task cannot also finish by t = 9 -> exactly one is served.
        seq = SequenceAwareExactSolver(CFG).solve([robot], [t0, t1])
        self.assertEqual(len(seq["unassigned_tasks"]), 1)


class TestOnlinePolicies(unittest.TestCase):
    def test_policies_return_only_valid_actions(self):
        from src.env.warehouse_env import WarehouseMRTAEnv
        for policy in (OnlineGreedyPolicy(CFG), RandomMaskedPolicy(7, CFG)):
            env = WarehouseMRTAEnv(CommunicationChannel("L2_Medium", CFG), CFG)
            obs, info = env.reset(seed=5)
            policy.bind(env)
            masks = info["action_masks"]
            for _ in range(30):
                actions = policy.select_actions(obs, masks)
                for rid, action in actions.items():
                    self.assertTrue(masks[rid][action])
                obs, _, done, step_info = env.step(actions)
                masks = step_info["action_masks"]
                if done:
                    break

    def test_l0_is_silent_for_online_policies(self):
        res = evaluate_online_policy_on_env(OnlineGreedyPolicy(CFG), "L0_None", 2, CFG)
        self.assertEqual(res["messages_sent"], 0)


class TestAllocationClockFix(unittest.TestCase):
    def test_full_communication_auction_equals_centralized_greedy(self):
        """With the clock fix, an auction that hears every bid is task-wise centralized greedy."""
        for seed in range(1, 11):
            robots, tasks = generate_scenario(seed)
            cg = CentralizedGreedyAllocator(CFG).solve(copy.deepcopy(robots), copy.deepcopy(tasks))
            ssi = AuctionAllocator(CommunicationChannel("L4_Full", CFG), CFG).solve(
                copy.deepcopy(robots), copy.deepcopy(tasks)
            )
            self.assertAlmostEqual(cg["assignment_objective_cost"], ssi["assignment_objective_cost"], places=6)


class TestAccountingIdentity(unittest.TestCase):
    def test_every_task_is_completed_missed_or_unassigned_exactly_once(self):
        """Regression test: late completions must be counted as missed deadlines."""
        for seed in range(1, 11):
            robots, tasks = generate_scenario(seed)
            plan = CentralizedGreedyAllocator(CFG).solve(robots, tasks)["assignments"]
            phys = _replay(plan, seed)
            total = phys["physically_completed_count"] + phys["missed_deadlines_count"] + phys["unassigned_count"]
            self.assertEqual(total, CFG.n_tasks, f"seed {seed}")

    def test_identity_for_online_policy(self):
        # evaluate_online_policy_on_env reports the same three counters
        for seed in range(1, 6):
            r = evaluate_online_policy_on_env(RandomMaskedPolicy(seed, CFG), "L1_Low", seed, CFG)
            self.assertEqual(
                r["physically_completed_count"] + r["missed_deadlines_count"] + r["unassigned_count"], CFG.n_tasks
            )


if __name__ == "__main__":
    unittest.main()
