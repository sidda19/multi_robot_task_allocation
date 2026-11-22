import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from typing import List, Dict, Tuple, Any
from src.env.entities import Robot, Task
from config import DEFAULT_CONFIG

class ExactILPSolver:
    """
    Solves the Multi-Robot Task Allocation problem to global theoretical optimality
    using Mixed-Integer Linear Programming (MILP via SciPy HiGHS).
    
    Mathematical Formulation (Ideation Report Eq. 1-7):
        min_{x, u} sum_{i=1}^N sum_{j=1}^M c_ij x_ij + P sum_{j=1}^M u_j
    subject to:
        sum_{i=1}^N x_ij + u_j == 1       for all j = 1..M (single assignment or unassigned)
        sum_{j=1}^M x_ij <= K_i           for all i = 1..N (capacity constraint)
        x_ij == 0                         if q_j not in Q_i (capability constraint)
        x_ij == 0                         if C_ij > D_j (deadline constraint)
        x_ij in {0, 1}, u_j in {0, 1}
    """
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config

    def solve(self, robots: List[Robot], tasks: List[Task]) -> Dict[str, Any]:
        N = len(robots)
        M = len(tasks)

        # Decision variables:
        # x_ij for i in [0..N-1], j in [0..M-1] -> index i * M + j
        # u_j for j in [0..M-1] -> index N * M + j
        num_vars = N * M + M

        c = np.zeros(num_vars)
        upper_bounds = np.ones(num_vars)
        lower_bounds = np.zeros(num_vars)

        # Fill objective costs
        cost_matrix = np.zeros((N, M))
        for i, robot in enumerate(robots):
            for j, task in enumerate(tasks):
                var_idx = i * M + j
                # Check capability
                feasible_cap = task.capability_req in robot.capabilities
                # Check deadline feasibility
                est_completion = robot.estimated_completion_time(task, current_step=0)
                feasible_deadline = est_completion <= task.deadline

                if not feasible_cap or not feasible_deadline:
                    # Infeasible assignment: lock to 0
                    upper_bounds[var_idx] = 0.0
                    cost_matrix[i, j] = 1e6
                else:
                    cij = robot.estimated_cost_to_task(task, self.config.alpha, self.config.beta)
                    c[var_idx] = cij
                    cost_matrix[i, j] = cij

        # Unassigned task penalty
        for j in range(M):
            u_idx = N * M + j
            c[u_idx] = self.config.penalty_unassigned

        # Constraint 1: sum_i x_ij + u_j == 1 for each task j
        A_eq = []
        b_eq = []
        for j in range(M):
            row = np.zeros(num_vars)
            for i in range(N):
                row[i * M + j] = 1.0
            row[N * M + j] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)

        # Constraint 2: sum_j x_ij <= K_i for each robot i
        A_ub = []
        b_ub = []
        for i, robot in enumerate(robots):
            row = np.zeros(num_vars)
            for j in range(M):
                row[i * M + j] = 1.0
            A_ub.append(row)
            b_ub.append(float(robot.capacity_limit))

        A_all = np.vstack([np.array(A_eq), np.array(A_ub)])
        lhs = np.concatenate([np.array(b_eq), np.zeros(N)])
        rhs = np.concatenate([np.array(b_eq), np.array(b_ub)])

        constraints = LinearConstraint(A_all, lhs, rhs)
        bounds = Bounds(lower_bounds, upper_bounds)
        integrality = np.ones(num_vars)  # All binary integer variables

        res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)

        assignments: Dict[int, List[int]] = {i: [] for i in range(N)}
        unassigned: List[int] = []
        assignment_cost = 0.0

        if res.success:
            sol = np.round(res.x)
            for i in range(N):
                for j in range(M):
                    if sol[i * M + j] == 1.0:
                        assignments[i].append(tasks[j].id)
                        assignment_cost += cost_matrix[i, j]
            for j in range(M):
                if sol[N * M + j] == 1.0:
                    unassigned.append(tasks[j].id)

            unassigned_penalty = len(unassigned) * self.config.penalty_unassigned
            conflict_penalty = 0.0  # ILP has zero conflicts by formulation
            total_cost = assignment_cost + unassigned_penalty + conflict_penalty
        else:
            assignment_cost = float("inf")
            unassigned_penalty = float("inf")
            conflict_penalty = 0.0
            total_cost = float("inf")

        runtime_ms = getattr(res, "execution_time", 0.0) * 1000 if hasattr(res, "execution_time") else 0.0

        return {
            "success": res.success,
            "total_cost": total_cost,
            "optimal_cost": total_cost,
            "assignment_objective_cost": total_cost,
            "assignment_cost": assignment_cost,
            "unassigned_penalty": unassigned_penalty,
            "conflict_penalty": conflict_penalty,
            "assignments": assignments,
            "unassigned_tasks": unassigned,
            "conflict_count": 0,
            "messages_sent": 0,
            "assignment_rate": (M - len(unassigned)) / M if M > 0 else 0.0,
            "completion_rate": (M - len(unassigned)) / M if M > 0 else 0.0,
            "runtime_ms": runtime_ms
        }
