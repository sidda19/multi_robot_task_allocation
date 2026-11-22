"""
Configuration parameters for Multi-Robot Task Allocation Under Limited Communication.
Addresses TA Feedback with concrete numerical parameters for small-scale, precise grading.
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SimulationConfig:
    # Environment Scale
    grid_width: int = 10
    grid_height: int = 10
    n_robots: int = 3          # Small scale: 3 robots (matched to 3 group members)
    n_tasks: int = 10          # Small scale: 10 tasks
    max_steps: int = 60        # Max episode steps
    
    # Robot Parameters
    robot_capacity: int = 3    # Max tasks handled in episode per robot
    robot_battery_max: float = 100.0
    battery_cost_per_unit_dist: float = 1.0
    
    # Task Parameters
    task_duration_min: int = 2
    task_duration_max: int = 4
    deadline_min: int = 20     # Minimum deadline steps
    deadline_max: int = 45     # Maximum deadline steps
    
    # Cost & Objective Weights
    alpha: float = 1.0         # Travel distance weight
    beta: float = 0.5          # Resource/effort weight
    penalty_unassigned: float = 50.0  # Penalty for dropped/unassigned task
    penalty_conflict: float = 10.0    # Penalty per losing duplicate robot-task claim
    
    # Communication Levels (Quantitative specification addressing TA Feedback #1)
    # k: communication period (sync every k steps)
    # budget: total messages allowed per robot per episode
    COMM_LEVELS: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.COMM_LEVELS is None:
            self.COMM_LEVELS = {
                "L0_None": {
                    "k": float("inf"),      # No exchange
                    "budget": 0,
                    "description": "No communication; robots use only initial local info"
                },
                "L1_Low": {
                    "k": 10,                 # Exchange every 10 steps
                    "budget": 6,             # Strict message budget
                    "description": "Low communication; infrequent periodic exchange"
                },
                "L2_Medium": {
                    "k": 5,                  # Exchange every 5 steps
                    "budget": 15,
                    "description": "Medium communication; regular interval exchange"
                },
                "L3_High": {
                    "k": 1,                  # Exchange every single step
                    "budget": 100,           # High message allowance
                    "description": "High communication; frequent exchange each step"
                },
                "L4_Full": {
                    "k": 0,                  # Instantaneous shared global state
                    "budget": float("inf"),
                    "description": "Instantaneous, unrestricted delivery of explicitly transmitted messages; not omniscient global state"
                }
            }

DEFAULT_CONFIG = SimulationConfig()
