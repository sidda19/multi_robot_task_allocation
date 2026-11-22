"""
Multi-Robot Task Allocation Under Limited Communication (MRTA-LC)
Common-seed experimental runner.

Static allocation quality is compared against Exact ILP using the same objective:
estimated assignment cost + unassigned-task penalty. Conflict overhead is reported
separately. All heuristic plans are replayed in the same physical simulator.
MARL is an online policy and is not assigned a static ILP optimality gap.
"""
import os
import sys
import time
import json
import copy
import hashlib
import platform
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

_libs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "libs"))
if os.path.exists(_libs_path) and _libs_path not in sys.path:
    sys.path.insert(0, _libs_path)

from src.env.entities import Location, Task, Robot
from src.env.warehouse_env import WarehouseMRTAEnv
from src.communication.channel import CommunicationChannel
from src.baselines.exact_ilp import ExactILPSolver
from src.baselines.exact_ilp_sequenced import SequenceAwareExactSolver
from src.marl.shared_dqfd import SharedDQNController, train_shared_dqfd
from src.baselines.online_policies import OnlineGreedyPolicy, RandomMaskedPolicy
from src.baselines.centralized_greedy import CentralizedGreedyAllocator
from src.baselines.distributed_greedy import DistributedGreedyAllocator
from src.baselines.auction import AuctionAllocator
from src.marl.marl_agent import CooperativeMARLController
from src.evaluation.metrics import compute_metrics
from config import DEFAULT_CONFIG, SimulationConfig


def generate_scenario(seed: int, config: SimulationConfig = DEFAULT_CONFIG) -> Tuple[List[Robot], List[Task]]:
    """Generate one deterministic scenario shared by every method."""
    rng = np.random.RandomState(seed)
    robots = []
    for i in range(config.n_robots):
        pos = Location(
            x=int(rng.randint(0, config.grid_width)),
            y=int(rng.randint(0, config.grid_height))
        )
        cap = {"A"} if i == 0 else ({"B"} if i == 1 else {"A", "B"})
        robots.append(Robot(
            id=i, position=pos, capabilities=cap,
            battery=float(config.robot_battery_max),
            capacity_limit=int(config.robot_capacity)
        ))

    tasks = []
    for j in range(config.n_tasks):
        pickup = Location(
            x=int(rng.randint(0, config.grid_width)),
            y=int(rng.randint(0, config.grid_height))
        )
        delivery = Location(
            x=int(rng.randint(0, config.grid_width)),
            y=int(rng.randint(0, config.grid_height))
        )
        req = "A" if j % 2 == 0 else "B"
        duration = int(rng.randint(config.task_duration_min, config.task_duration_max + 1))
        deadline = int(rng.randint(config.deadline_min, config.deadline_max + 1))
        tasks.append(Task(
            id=j, pickup=pickup, delivery=delivery, capability_req=req,
            release_time=0, duration=duration, deadline=deadline
        ))
    return robots, tasks


def _missed_deadline(task: Task) -> bool:
    """Assigned tasks that are incomplete OR completed after their deadline."""
    return (
        task.assigned_robot is not None
        and (
            not task.completed
            or task.completion_time is None
            or task.completion_time > task.deadline
        )
    )


def evaluate_marl_on_env(
    controller: CooperativeMARLController,
    comm_level: str,
    seed: int,
    config: SimulationConfig
) -> Dict[str, Any]:
    """Evaluate MARL as an integrated online policy + physical episode."""
    channel = CommunicationChannel(comm_level, config)
    env = WarehouseMRTAEnv(channel, config)
    obs, info = env.reset(seed=seed)
    masks = info["action_masks"]
    if hasattr(controller, "bind"):          # reference policies read the robot's own cost estimate
        controller.bind(env)

    start_time = time.perf_counter()
    done = False
    total_conflicts = 0
    while not done:
        actions = controller.select_actions(obs, masks, training=False)
        next_obs, _, done, step_info = env.step(actions)
        total_conflicts += step_info.get("conflicts_this_step", 0)
        obs = next_obs
        masks = step_info["action_masks"]
    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    on_time_completed = [
        t for t in env.tasks
        if t.completed and t.completion_time is not None and t.completion_time <= t.deadline
    ]
    missed_deadlines = [t for t in env.tasks if _missed_deadline(t)]
    assigned_tasks = [t for t in env.tasks if t.assigned_robot is not None]
    unassigned_tasks = [t for t in env.tasks if t.assigned_robot is None]
    travel = float(sum(r.total_travel_distance for r in env.robots))
    unassigned_penalty = len(unassigned_tasks) * config.penalty_unassigned
    missed_penalty = len(missed_deadlines) * config.penalty_unassigned
    conflict_penalty = total_conflicts * config.penalty_conflict
    physical_cost = travel + unassigned_penalty + missed_penalty + conflict_penalty

    return {
        "total_cost": physical_cost,
        "physical_execution_cost": physical_cost,
        "allocation_conflict_penalty": 0.0,
        "assignment_objective_cost": None,
        "static_assignment_cost": None,
        "assignment_cost": travel,
        "assigned_count": len(assigned_tasks),
        "assignment_rate": len(assigned_tasks) / float(config.n_tasks),
        "physically_completed_count": len(on_time_completed),
        "physical_completion_rate": len(on_time_completed) / float(config.n_tasks),
        "completion_rate": len(on_time_completed) / float(config.n_tasks),
        "missed_deadlines_count": len(missed_deadlines),
        "unassigned_count": len(unassigned_tasks),
        "conflict_count": total_conflicts,
        "allocation_conflict_count": 0,
        "physical_conflict_count": total_conflicts,
        "allocation_unassigned_count": None,
        "physical_unassigned_count": len(unassigned_tasks),
        "messages_sent": channel.total_messages,
        "travel_distance": travel,
        "runtime_ms": runtime_ms,
        "allocation_runtime_ms": None,
        "physical_simulation_runtime_ms": runtime_ms
    }


# Any online policy exposing select_actions(obs, masks, training=False) can be evaluated here.
evaluate_online_policy_on_env = evaluate_marl_on_env


def execute_static_allocation_physically(
    assignments: Dict[int, List[int]],
    comm_channel: CommunicationChannel,
    seed: int,
    config: SimulationConfig
) -> Dict[str, Any]:
    """Replay a fixed static plan in the common physical simulator."""
    env = WarehouseMRTAEnv(comm_channel, config)
    env.reset(seed=seed)
    start = time.perf_counter()
    result = env.run_physical_execution(assignments)
    result["physical_simulation_runtime_ms"] = (time.perf_counter() - start) * 1000.0
    return result


def _checkpoint_manifest(model_dir: str, n_robots: int) -> Dict[str, Any]:
    manifest = {}
    for robot_id in range(n_robots):
        path = os.path.join(model_dir, f"agent_{robot_id}.pt")
        if os.path.isfile(path):
            with open(path, "rb") as checkpoint_file:
                digest = hashlib.sha256(checkpoint_file.read()).hexdigest()
            manifest[f"agent_{robot_id}.pt"] = {
                "sha256": digest,
                "size_bytes": os.path.getsize(path)
            }
    return manifest


def run_all_experiments(
    seeds: List[int] = None,
    config: SimulationConfig = DEFAULT_CONFIG,
    model_dir: str = "models/idqn_model",
    results_dir: str = "results"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if seeds is None:
        seeds = list(range(1, 11))

    print("=" * 80)
    print("Multi-Robot Task Allocation Under Limited Communication (MRTA-LC)")
    print("Corrected common-objective and common-physical-execution pipeline")
    print(f"Seeds: {seeds}")
    print(f"Robots: {config.n_robots}, Tasks: {config.n_tasks}, Max Steps: {config.max_steps}")
    print("=" * 80)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    # Load the supplied checkpoint or train a deterministic replacement.
    obs_dim = 7 + config.n_tasks * 8
    action_dim = config.n_tasks + 1
    marl_controller = CooperativeMARLController(
        n_robots=config.n_robots, obs_dim=obs_dim,
        action_dim=action_dim, hidden_dim=64
    )
    checkpoint_paths = [
        os.path.join(model_dir, f"agent_{i}.pt") for i in range(config.n_robots)
    ]
    if all(os.path.isfile(path) for path in checkpoint_paths):
        marl_controller.load(model_dir)
        metadata_file = os.path.join(model_dir, "training_metadata.json")
        checkpoint_source = (
            "trained_checkpoint_with_metadata"
            if os.path.isfile(metadata_file)
            else "supplied_checkpoint_provenance_unknown"
        )
        print(f"Loaded trained MARL IDQN models from: {model_dir}")
    else:
        print("Complete MARL checkpoint not found; training a seeded 120-episode model.")
        from src.marl.train_marl import train_idqn
        train_result = train_idqn(
            n_episodes=120, seed=2026, save_dir=model_dir,
            config=config, verbose=True
        )
        marl_controller = train_result["controller"]
        checkpoint_source = "trained_during_experiment_run"

    training_metadata_path = os.path.join(model_dir, "training_metadata.json")
    training_metadata = None
    if os.path.isfile(training_metadata_path):
        with open(training_metadata_path, "r", encoding="utf-8") as f:
            training_metadata = json.load(f)

    shared_dir = os.path.join(os.path.dirname(model_dir.rstrip('/\\')) or '.', "shared_dqfd")
    shared_controller = SharedDQNController(config)
    if os.path.isfile(os.path.join(shared_dir, "shared_dqfd.pt")):
        shared_controller.load(shared_dir)
        print(f"Loaded shared-parameter MARL checkpoint from: {shared_dir}")
    else:
        print("Shared MARL checkpoint not found; training it (about 4 minutes).")
        shared_controller = train_shared_dqfd(save_dir=shared_dir, config=config)["controller"]
    shared_meta = None
    if os.path.isfile(os.path.join(shared_dir, "training_metadata.json")):
        with open(os.path.join(shared_dir, "training_metadata.json"), "r", encoding="utf-8") as f:
            shared_meta = json.load(f)

    config_dict = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "grid_width": config.grid_width,
        "grid_height": config.grid_height,
        "n_robots": config.n_robots,
        "n_tasks": config.n_tasks,
        "max_steps": config.max_steps,
        "robot_capacity": config.robot_capacity,
        "robot_battery_max": config.robot_battery_max,
        "battery_cost_per_unit_dist": config.battery_cost_per_unit_dist,
        "alpha": config.alpha,
        "beta": config.beta,
        "penalty_unassigned": config.penalty_unassigned,
        "penalty_conflict": config.penalty_conflict,
        "deadline_min": config.deadline_min,
        "deadline_max": config.deadline_max,
        "task_duration_min": config.task_duration_min,
        "task_duration_max": config.task_duration_max,
        "seeds": list(seeds),
        "comm_levels": {
            key: {
                "k": ("never" if value["k"] == float("inf") else value["k"]),
                "budget": ("unlimited" if value["budget"] == float("inf") else value["budget"]),
                "description": value["description"]
            }
            for key, value in config.COMM_LEVELS.items()
        },
        "marl_checkpoint_source": checkpoint_source,
        "marl_training": {
            "algorithm": "Independent DQN (IDQN)",
            "episodes": 120,
            "communication_level": "L2_Medium",
            "seed": (
                training_metadata.get("seed")
                if training_metadata is not None
                else (2026 if checkpoint_source == "trained_during_experiment_run" else None)
            ),
            "hidden_dim": 64,
            "learning_rate": 1e-3,
            "gamma": 0.95,
            "batch_size": 32
        },
        "marl_checkpoint_sha256": _checkpoint_manifest(model_dir, config.n_robots),
        "marl_checkpoint_training_metadata": training_metadata,
        "shared_marl_training_metadata": shared_meta
    }
    with open(os.path.join(results_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, allow_nan=False)

    comm_levels = ["L0_None", "L1_Low", "L2_Medium", "L3_High", "L4_Full"]
    records = []
    experiment_start = time.perf_counter()

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"[{seed_idx:2d}/{len(seeds)}] Evaluating Scenario Seed {seed}...")
        robots_init, tasks_init = generate_scenario(seed, config)

        # 1. Exact ILP: offline static objective reference, no physical replay.
        ilp_start = time.perf_counter()
        ilp_res = ExactILPSolver(config).solve(
            copy.deepcopy(robots_init), copy.deepcopy(tasks_init)
        )
        ilp_runtime_ms = (time.perf_counter() - ilp_start) * 1000.0
        if not ilp_res.get("success", False):
            raise RuntimeError(f"Exact ILP failed for scenario seed {seed}")
        optimal_cost = float(ilp_res["optimal_cost"])
        ilp_record = {
            "total_cost": None,
            "physical_execution_cost": None,
            "allocation_conflict_penalty": 0.0,
            "assignment_objective_cost": ilp_res["assignment_objective_cost"],
            "static_assignment_cost": ilp_res["assignment_cost"],
            "assignment_cost": None,
            "assigned_count": config.n_tasks - len(ilp_res["unassigned_tasks"]),
            "assignment_rate": ilp_res["assignment_rate"],
            "physically_completed_count": 0,
            "physical_completion_rate": None,
            "completion_rate": ilp_res["assignment_rate"],
            "missed_deadlines_count": 0,
            "unassigned_count": len(ilp_res["unassigned_tasks"]),
            "conflict_count": 0,
            "allocation_conflict_count": 0,
            "physical_conflict_count": 0,
            "allocation_unassigned_count": len(ilp_res["unassigned_tasks"]),
            "physical_unassigned_count": None,
            "messages_sent": 0,
            "travel_distance": None,
            "runtime_ms": ilp_runtime_ms,
            "allocation_runtime_ms": ilp_runtime_ms,
            "physical_simulation_runtime_ms": None
        }
        records.append(compute_metrics(
            "Exact ILP (assignment)", "N/A (Offline)", ilp_record,
            optimal_cost, seed=seed
        ).to_dict())

        # 1b. Sequence-aware exact reference (set-partitioning ILP). It is the true
        # optimum of the physical cost for this simulator and is replayed to prove it.
        seq_res = SequenceAwareExactSolver(config).solve(
            copy.deepcopy(robots_init), copy.deepcopy(tasks_init)
        )
        seq_phys = execute_static_allocation_physically(
            seq_res["assignments"], CommunicationChannel("L4_Full", config), seed, config
        )
        seq_opt_cost = float(seq_phys["total_physical_cost"])
        assert abs(seq_opt_cost - seq_res["optimal_cost"]) < 1e-6, "sequence-aware plan must replay exactly"
        seq_record = {
            "total_cost": seq_opt_cost,
            "physical_execution_cost": seq_opt_cost,
            "allocation_conflict_penalty": 0.0,
            "assignment_objective_cost": None,
            "static_assignment_cost": None,
            "assignment_cost": seq_phys["total_travel_distance"],
            "assigned_count": sum(len(v) for v in seq_res["assignments"].values()),
            "assignment_rate": seq_res["assignment_rate"],
            "physically_completed_count": seq_phys["physically_completed_count"],
            "physical_completion_rate": seq_phys["physical_completion_rate"],
            "completion_rate": seq_phys["physical_completion_rate"],
            "missed_deadlines_count": seq_phys["missed_deadlines_count"],
            "unassigned_count": seq_phys["unassigned_count"],
            "conflict_count": seq_phys["physical_conflicts"],
            "allocation_conflict_count": 0,
            "physical_conflict_count": seq_phys["physical_conflicts"],
            "allocation_unassigned_count": len(seq_res["unassigned_tasks"]),
            "physical_unassigned_count": seq_phys["unassigned_count"],
            "messages_sent": 0,
            "travel_distance": seq_phys["total_travel_distance"],
            "runtime_ms": seq_res["runtime_ms"] + seq_phys["physical_simulation_runtime_ms"],
            "allocation_runtime_ms": seq_res["runtime_ms"],
            "physical_simulation_runtime_ms": seq_phys["physical_simulation_runtime_ms"]
        }
        records.append(compute_metrics(
            "Exact ILP (sequence-aware)", "N/A (Offline)", seq_record,
            optimal_cost, seed=seed, seq_opt_reference_cost=seq_opt_cost
        ).to_dict())

        # 2. Centralized Greedy.
        alloc_start = time.perf_counter()
        cg_res = CentralizedGreedyAllocator(config).solve(
            copy.deepcopy(robots_init), copy.deepcopy(tasks_init)
        )
        cg_alloc_ms = (time.perf_counter() - alloc_start) * 1000.0
        cg_phys = execute_static_allocation_physically(
            cg_res["assignments"], CommunicationChannel("L4_Full", config), seed, config
        )
        cg_physical_cost = float(cg_phys["total_physical_cost"])
        cg_record = {
            "total_cost": cg_physical_cost,
            "physical_execution_cost": cg_physical_cost,
            "allocation_conflict_penalty": 0.0,
            "assignment_objective_cost": cg_res["assignment_objective_cost"],
            "static_assignment_cost": cg_res["assignment_cost"],
            "assignment_cost": cg_phys["total_travel_distance"],
            "assigned_count": sum(len(v) for v in cg_res["assignments"].values()),
            "assignment_rate": cg_res["assignment_rate"],
            "physically_completed_count": cg_phys["physically_completed_count"],
            "physical_completion_rate": cg_phys["physical_completion_rate"],
            "completion_rate": cg_phys["physical_completion_rate"],
            "missed_deadlines_count": cg_phys["missed_deadlines_count"],
            "unassigned_count": cg_phys["unassigned_count"],
            "conflict_count": cg_res["conflict_count"] + cg_phys["physical_conflicts"],
            "allocation_conflict_count": cg_res["conflict_count"],
            "physical_conflict_count": cg_phys["physical_conflicts"],
            "allocation_unassigned_count": len(cg_res["unassigned_tasks"]),
            "physical_unassigned_count": cg_phys["unassigned_count"],
            "messages_sent": cg_res["messages_sent"],
            "travel_distance": cg_phys["total_travel_distance"],
            "runtime_ms": cg_alloc_ms + cg_phys["physical_simulation_runtime_ms"],
            "allocation_runtime_ms": cg_alloc_ms,
            "physical_simulation_runtime_ms": cg_phys["physical_simulation_runtime_ms"]
        }
        records.append(compute_metrics(
            "Centralized Greedy", "Global (Full Info)", cg_record,
            optimal_cost, seed=seed, seq_opt_reference_cost=seq_opt_cost
        ).to_dict())

        # 3. Distributed Greedy across communication levels.
        for comm in comm_levels:
            alloc_channel = CommunicationChannel(comm, config)
            alloc_start = time.perf_counter()
            dg_res = DistributedGreedyAllocator(alloc_channel, config).solve(
                copy.deepcopy(robots_init), copy.deepcopy(tasks_init)
            )
            dg_alloc_ms = (time.perf_counter() - alloc_start) * 1000.0
            dg_phys = execute_static_allocation_physically(
                dg_res["assignments"], CommunicationChannel(comm, config), seed, config
            )
            alloc_conflict_penalty = dg_res["conflict_count"] * config.penalty_conflict
            physical_cost = float(dg_phys["total_physical_cost"])
            end_to_end_cost = physical_cost + alloc_conflict_penalty
            dg_record = {
                "total_cost": end_to_end_cost,
                "physical_execution_cost": physical_cost,
                "allocation_conflict_penalty": alloc_conflict_penalty,
                "assignment_objective_cost": dg_res["assignment_objective_cost"],
                "static_assignment_cost": dg_res["assignment_cost"],
                "assignment_cost": dg_phys["total_travel_distance"],
                "assigned_count": sum(len(v) for v in dg_res["assignments"].values()),
                "assignment_rate": dg_res["assignment_rate"],
                "physically_completed_count": dg_phys["physically_completed_count"],
                "physical_completion_rate": dg_phys["physical_completion_rate"],
                "completion_rate": dg_phys["physical_completion_rate"],
                "missed_deadlines_count": dg_phys["missed_deadlines_count"],
                "unassigned_count": dg_phys["unassigned_count"],
                "conflict_count": dg_res["conflict_count"] + dg_phys["physical_conflicts"],
                "allocation_conflict_count": dg_res["conflict_count"],
                "physical_conflict_count": dg_phys["physical_conflicts"],
                "allocation_unassigned_count": len(dg_res["unassigned_tasks"]),
                "physical_unassigned_count": dg_phys["unassigned_count"],
                "messages_sent": dg_res["messages_sent"],
                "travel_distance": dg_phys["total_travel_distance"],
                "runtime_ms": dg_alloc_ms + dg_phys["physical_simulation_runtime_ms"],
                "allocation_runtime_ms": dg_alloc_ms,
                "physical_simulation_runtime_ms": dg_phys["physical_simulation_runtime_ms"]
            }
            records.append(compute_metrics(
                "Distributed Greedy", comm, dg_record, optimal_cost, seed=seed,
                seq_opt_reference_cost=seq_opt_cost
            ).to_dict())

        # 4. SSI Auction across communication levels.
        for comm in comm_levels:
            alloc_channel = CommunicationChannel(comm, config)
            alloc_start = time.perf_counter()
            auc_res = AuctionAllocator(alloc_channel, config).solve(
                copy.deepcopy(robots_init), copy.deepcopy(tasks_init)
            )
            auc_alloc_ms = (time.perf_counter() - alloc_start) * 1000.0
            auc_phys = execute_static_allocation_physically(
                auc_res["assignments"], CommunicationChannel(comm, config), seed, config
            )
            alloc_conflict_penalty = auc_res["conflict_count"] * config.penalty_conflict
            physical_cost = float(auc_phys["total_physical_cost"])
            end_to_end_cost = physical_cost + alloc_conflict_penalty
            auc_record = {
                "total_cost": end_to_end_cost,
                "physical_execution_cost": physical_cost,
                "allocation_conflict_penalty": alloc_conflict_penalty,
                "assignment_objective_cost": auc_res["assignment_objective_cost"],
                "static_assignment_cost": auc_res["assignment_cost"],
                "assignment_cost": auc_phys["total_travel_distance"],
                "assigned_count": sum(len(v) for v in auc_res["assignments"].values()),
                "assignment_rate": auc_res["assignment_rate"],
                "physically_completed_count": auc_phys["physically_completed_count"],
                "physical_completion_rate": auc_phys["physical_completion_rate"],
                "completion_rate": auc_phys["physical_completion_rate"],
                "missed_deadlines_count": auc_phys["missed_deadlines_count"],
                "unassigned_count": auc_phys["unassigned_count"],
                "conflict_count": auc_res["conflict_count"] + auc_phys["physical_conflicts"],
                "allocation_conflict_count": auc_res["conflict_count"],
                "physical_conflict_count": auc_phys["physical_conflicts"],
                "allocation_unassigned_count": len(auc_res["unassigned_tasks"]),
                "physical_unassigned_count": auc_phys["unassigned_count"],
                "messages_sent": auc_res["messages_sent"],
                "travel_distance": auc_phys["total_travel_distance"],
                "runtime_ms": auc_alloc_ms + auc_phys["physical_simulation_runtime_ms"],
                "allocation_runtime_ms": auc_alloc_ms,
                "physical_simulation_runtime_ms": auc_phys["physical_simulation_runtime_ms"]
            }
            records.append(compute_metrics(
                "SSI Auction", comm, auc_record, optimal_cost, seed=seed,
                seq_opt_reference_cost=seq_opt_cost
            ).to_dict())

        # 5. Cooperative MARL, evaluated as an online physical policy.
        for comm in comm_levels:
            marl_res = evaluate_marl_on_env(marl_controller, comm, seed, config)
            records.append(compute_metrics(
                "Cooperative MARL (IDQN)", comm, marl_res, optimal_cost, seed=seed,
                seq_opt_reference_cost=seq_opt_cost
            ).to_dict())

        # 5b. Cooperative MARL, shared-parameter policy trained from demonstrations + team reward.
        for comm in comm_levels:
            shared_res = evaluate_marl_on_env(shared_controller, comm, seed, config)
            records.append(compute_metrics(
                "Cooperative MARL (shared DQfD)", comm, shared_res, optimal_cost, seed=seed,
                seq_opt_reference_cost=seq_opt_cost
            ).to_dict())

        # 6. Reference online policies in the SAME environment/channel as MARL.
        #    They show what the environment + action mask alone already achieve.
        for comm in comm_levels:
            greedy_res = evaluate_online_policy_on_env(OnlineGreedyPolicy(config), comm, seed, config)
            records.append(compute_metrics(
                "Online Distributed Greedy", comm, greedy_res, optimal_cost, seed=seed,
                seq_opt_reference_cost=seq_opt_cost
            ).to_dict())
        for comm in comm_levels:
            conv_res = evaluate_online_policy_on_env(
                OnlineGreedyPolicy(config, tie_break="convention"), comm, seed, config
            )
            records.append(compute_metrics(
                "Online Greedy (no-message convention)", comm, conv_res, optimal_cost, seed=seed,
                seq_opt_reference_cost=seq_opt_cost
            ).to_dict())
        for comm in comm_levels:
            random_res = evaluate_online_policy_on_env(
                RandomMaskedPolicy(seed=1000 * seed + comm_levels.index(comm), config=config),
                comm, seed, config
            )
            records.append(compute_metrics(
                "Random Masked Policy", comm, random_res, optimal_cost, seed=seed,
                seq_opt_reference_cost=seq_opt_cost
            ).to_dict())

    total_experiment_time = time.perf_counter() - experiment_start
    print(f"\nAll experiments successfully completed in {total_experiment_time:.2f} seconds!")

    df_raw = pd.DataFrame(records)
    raw_path = os.path.join(results_dir, "experiment_results.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"Detailed raw results saved to: {raw_path}")

    numeric_cols = [
        "total_cost", "physical_execution_cost", "allocation_conflict_penalty",
        "assignment_cost", "assignment_objective_cost", "static_assignment_cost", "optimality_gap_%",
        "assignment_rate_%", "physical_completion_rate_%", "conflicts",
        "allocation_conflicts", "physical_conflicts", "allocation_unassigned_count",
        "physical_unassigned_count", "messages", "travel_distance", "runtime_ms",
        "allocation_runtime_ms", "physical_simulation_runtime_ms",
        "e2e_gap_vs_seq_opt_%", "seq_opt_reference_cost", "missed_deadlines", "physically_completed_count"
    ]
    agg_rows = []
    for (method, comm), group in df_raw.groupby(["method", "comm_level"], sort=False):
        row = {"method": method, "comm_level": comm}
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            if values.notna().any():
                row[f"{col}_mean"] = round(float(values.mean()), 3)
                row[f"{col}_std"] = round(float(values.std()), 3) if values.count() > 1 else 0.0
            else:
                row[f"{col}_mean"] = "N/A"
                row[f"{col}_std"] = "N/A"
        agg_rows.append(row)

    df_agg = pd.DataFrame(agg_rows)
    agg_path = os.path.join(results_dir, "aggregated_results.csv")
    df_agg.to_csv(agg_path, index=False)
    print(f"Aggregated summary results saved to: {agg_path}")
    return df_raw, df_agg


if __name__ == "__main__":
    run_all_experiments()
