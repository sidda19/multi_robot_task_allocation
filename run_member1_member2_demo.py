"""
Demo runner comparing Member 1 and Member 2 baselines across communication levels L0 to L4.
Prints a formatted summary table comparing Exact ILP, Centralized Greedy, Distributed Greedy, and Auction.
"""
import sys
import pandas as pd
from src.env.entities import Location, Task, Robot
from src.baselines.exact_ilp import ExactILPSolver
from src.baselines.centralized_greedy import CentralizedGreedyAllocator
from src.baselines.distributed_greedy import DistributedGreedyAllocator
from src.baselines.auction import AuctionAllocator
from src.communication.channel import CommunicationChannel
from src.evaluation.metrics import compute_metrics
from config import DEFAULT_CONFIG

def run_demo():
    print("=" * 80)
    print("Multi-Robot Task Allocation Under Limited Communication (MRTA-LC)")
    print("Member 1 & Member 2 Baselines Verification")
    print("=" * 80)

    # 3 Robots (Small scale for grading)
    robots = [
        Robot(id=0, position=Location(1, 1), capabilities={"A"}, capacity_limit=3),
        Robot(id=1, position=Location(2, 2), capabilities={"B"}, capacity_limit=3),
        Robot(id=2, position=Location(5, 5), capabilities={"A", "B"}, capacity_limit=3)
    ]

    # 6 Tasks with coordinates, capabilities, and deadlines
    tasks = [
        Task(id=0, pickup=Location(1, 2), delivery=Location(3, 3), capability_req="A", deadline=25),
        Task(id=1, pickup=Location(2, 1), delivery=Location(4, 2), capability_req="B", deadline=25),
        Task(id=2, pickup=Location(6, 6), delivery=Location(8, 8), capability_req="A", deadline=30),
        Task(id=3, pickup=Location(5, 4), delivery=Location(7, 5), capability_req="B", deadline=30),
        Task(id=4, pickup=Location(0, 3), delivery=Location(2, 5), capability_req="A", deadline=35),
        Task(id=5, pickup=Location(4, 1), delivery=Location(6, 3), capability_req="B", deadline=35)
    ]

    # 1. Member 1: Exact ILP Optimal Benchmark
    ilp_solver = ExactILPSolver(DEFAULT_CONFIG)
    ilp_res = ilp_solver.solve(robots, tasks)
    opt_cost = ilp_res["optimal_cost"]
    
    records = []
    
    ilp_metrics = compute_metrics("Exact ILP (Theoretical Optimal)", "N/A", ilp_res, opt_cost)
    records.append(ilp_metrics.to_dict())

    # 2. Member 1: Centralized Greedy
    cg_alloc = CentralizedGreedyAllocator(DEFAULT_CONFIG)
    cg_res = cg_alloc.solve(robots, tasks)
    cg_metrics = compute_metrics("Centralized Greedy", "Global (L4)", cg_res, opt_cost)
    records.append(cg_metrics.to_dict())

    # 3. Member 2: Distributed Greedy across L0-L4
    for lvl in ["L0_None", "L1_Low", "L2_Medium", "L3_High", "L4_Full"]:
        ch = CommunicationChannel(lvl, DEFAULT_CONFIG)
        dg = DistributedGreedyAllocator(ch, DEFAULT_CONFIG)
        dg_res = dg.solve(robots, tasks)
        metrics = compute_metrics("Distributed Greedy", lvl, dg_res, opt_cost)
        records.append(metrics.to_dict())

    # 4. Member 2: Auction Baseline across L0-L4
    for lvl in ["L0_None", "L1_Low", "L2_Medium", "L3_High", "L4_Full"]:
        ch = CommunicationChannel(lvl, DEFAULT_CONFIG)
        auc = AuctionAllocator(ch, DEFAULT_CONFIG)
        auc_res = auc.solve(robots, tasks)
        metrics = compute_metrics("Sequential Single-Item Auction", lvl, auc_res, opt_cost)
        records.append(metrics.to_dict())

    df = pd.DataFrame(records)
    print("\nBenchmark Summary Table:")
    print(df.to_string(index=False))
    print("\n" + "=" * 80)
    print("Verification successfully finished. All baselines operational!")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
