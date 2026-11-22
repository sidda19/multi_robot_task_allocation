import copy
import unittest

from config import DEFAULT_CONFIG as cfg
from run_experiments import generate_scenario, execute_static_allocation_physically, evaluate_marl_on_env
from src.baselines.auction import AuctionAllocator
from src.baselines.centralized_greedy import CentralizedGreedyAllocator
from src.baselines.distributed_greedy import DistributedGreedyAllocator
from src.baselines.online_policies import OnlineGreedyPolicy
from src.communication.channel import CommunicationChannel


class SequenceAwarePlannerTests(unittest.TestCase):
    def _replay(self, plan, seed, lvl="L4_Full"):
        return execute_static_allocation_physically(plan["assignments"], CommunicationChannel(lvl, cfg), seed, cfg)

    def test_centralized_greedy_plans_never_miss_deadlines(self):
        for seed in range(1, 11):
            robots, tasks = generate_scenario(seed)
            plan = CentralizedGreedyAllocator(cfg).solve(robots, tasks)
            self.assertEqual(self._replay(plan, seed)["missed_deadlines_count"], 0, seed)

    def test_distributed_greedy_and_auction_plans_never_miss_deadlines(self):
        for seed in range(1, 11):
            for lvl in ("L1_Low", "L2_Medium", "L4_Full"):
                robots, tasks = generate_scenario(seed)
                for cls in (DistributedGreedyAllocator, AuctionAllocator):
                    plan = cls(CommunicationChannel(lvl, cfg), cfg).solve(copy.deepcopy(robots), copy.deepcopy(tasks))
                    # conflicts are only possible when no message was delivered in time (L0 is excluded here)
                    self.assertEqual(self._replay(plan, seed, lvl)["missed_deadlines_count"], 0, (cls.__name__, lvl, seed))

    def test_legacy_flag_still_reproduces_old_behaviour(self):
        robots, tasks = generate_scenario(1)
        old = CentralizedGreedyAllocator(cfg, sequence_aware=False).solve(robots, tasks)
        self.assertGreater(self._replay(old, 1)["missed_deadlines_count"], 0)

    def test_convention_tie_break_reduces_l0_conflicts(self):
        tot = {"none": 0, "convention": 0}
        for seed in range(1, 11):
            for tb in tot:
                tot[tb] += evaluate_marl_on_env(OnlineGreedyPolicy(cfg, tie_break=tb), "L0_None", seed, cfg)["conflict_count"]
        self.assertLess(tot["convention"], tot["none"])


if __name__ == "__main__":
    unittest.main()
