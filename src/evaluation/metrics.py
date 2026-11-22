from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class EvaluationMetrics:
    method: str
    comm_level: str
    total_cost: Optional[float]
    assignment_cost: Optional[float]
    optimality_gap_percent: Optional[float]
    assignment_objective_cost: Optional[float]
    completion_rate: float
    assignment_rate: float
    physical_completion_rate: Optional[float]
    conflict_count: int
    messages_sent: int
    runtime_ms: float
    seed: int = 0
    assigned_count: int = 0
    physically_completed_count: int = 0
    unassigned_count: int = 0
    missed_deadline_count: int = 0
    travel_distance: Optional[float] = 0.0
    training_time_s: float = 0.0
    physical_execution_cost: Optional[float] = None
    allocation_conflict_penalty: float = 0.0
    allocation_runtime_ms: Optional[float] = None
    physical_simulation_runtime_ms: Optional[float] = None
    allocation_conflict_count: int = 0
    physical_conflict_count: int = 0
    allocation_unassigned_count: Optional[int] = None
    physical_unassigned_count: Optional[int] = None
    static_assignment_cost: Optional[float] = None
    seq_opt_reference_cost: Optional[float] = None
    e2e_gap_vs_seq_opt_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        phys_comp_val = (
            round(self.physical_completion_rate * 100, 1)
            if self.physical_completion_rate is not None else "N/A"
        )
        return {
            "seed": self.seed,
            "method": self.method,
            "comm_level": self.comm_level,
            # total_cost is the explicitly defined end-to-end cost for a method
            "total_cost": round(self.total_cost, 2) if self.total_cost is not None else "N/A",
            "end_to_end_total_cost": round(self.total_cost, 2) if self.total_cost is not None else "N/A",
            "physical_execution_cost": (
                round(self.physical_execution_cost, 2)
                if self.physical_execution_cost is not None else "N/A"
            ),
            "allocation_conflict_penalty": round(self.allocation_conflict_penalty, 2),
            "assignment_cost": round(self.assignment_cost, 2) if self.assignment_cost is not None else "N/A",
            "optimality_gap_%": round(self.optimality_gap_percent, 2) if self.optimality_gap_percent is not None else "N/A",
            "assignment_objective_cost": (
                round(self.assignment_objective_cost, 2)
                if self.assignment_objective_cost is not None else "N/A"
            ),
            "static_assignment_cost": (
                round(self.static_assignment_cost, 2)
                if self.static_assignment_cost is not None else "N/A"
            ),
            "seq_opt_reference_cost": (
                round(self.seq_opt_reference_cost, 2)
                if self.seq_opt_reference_cost is not None else "N/A"
            ),
            "e2e_gap_vs_seq_opt_%": (
                round(self.e2e_gap_vs_seq_opt_percent, 2)
                if self.e2e_gap_vs_seq_opt_percent is not None else "N/A"
            ),
            "assignment_rate_%": round(self.assignment_rate * 100, 1),
            "physical_completion_rate_%": phys_comp_val,
            "completion_rate_%": round(self.completion_rate * 100, 1),
            "assigned_count": int(self.assigned_count),
            "physically_completed_count": int(self.physically_completed_count),
            "unassigned_count": int(self.unassigned_count),
            "missed_deadlines": int(self.missed_deadline_count),
            "conflicts": int(self.conflict_count),
            "allocation_conflicts": int(self.allocation_conflict_count),
            "physical_conflicts": int(self.physical_conflict_count),
            "allocation_unassigned_count": (
                int(self.allocation_unassigned_count)
                if self.allocation_unassigned_count is not None else "N/A"
            ),
            "physical_unassigned_count": (
                int(self.physical_unassigned_count)
                if self.physical_unassigned_count is not None else "N/A"
            ),
            "messages": int(self.messages_sent),
            "travel_distance": round(self.travel_distance, 1) if self.travel_distance is not None else "N/A",
            "runtime_ms": round(self.runtime_ms, 3),
            "allocation_runtime_ms": (
                round(self.allocation_runtime_ms, 3)
                if self.allocation_runtime_ms is not None else "N/A"
            ),
            "physical_simulation_runtime_ms": (
                round(self.physical_simulation_runtime_ms, 3)
                if self.physical_simulation_runtime_ms is not None else "N/A"
            ),
            "training_time_s": round(self.training_time_s, 2)
        }


def compute_metrics(
    method_name: str,
    comm_level: str,
    result: Dict[str, Any],
    optimal_cost: Optional[float],
    seed: int = 0,
    training_time_s: float = 0.0,
    seq_opt_reference_cost: Optional[float] = None
) -> EvaluationMetrics:
    total_cost = result.get("total_cost")
    objective_cost = result.get("assignment_objective_cost")
    gap = None

    # Only static allocator objectives (assignment estimate + unassigned penalty)
    # are compared with the static Exact ILP objective. Conflict penalties and
    # physical execution costs are not part of this optimality-gap calculation.
    if objective_cost is not None and optimal_cost is not None and optimal_cost > 0:
        gap = ((float(objective_cost) - float(optimal_cost)) / float(optimal_cost)) * 100.0

    # Physical (end-to-end) gap against the sequence-aware exact reference. It is
    # defined for every method that is executed physically, including MARL.
    e2e_gap = None
    if total_cost is not None and seq_opt_reference_cost is not None and seq_opt_reference_cost > 0:
        e2e_gap = (float(total_cost) - float(seq_opt_reference_cost)) / float(seq_opt_reference_cost) * 100.0

    assign_rate = float(result.get("assignment_rate", result.get("completion_rate", 0.0)))
    phys_rate = result.get("physical_completion_rate")
    phys_rate = float(phys_rate) if phys_rate is not None else None
    comp_rate = phys_rate if phys_rate is not None else assign_rate

    return EvaluationMetrics(
        seed=seed,
        method=method_name,
        comm_level=comm_level,
        total_cost=float(total_cost) if total_cost is not None else None,
        assignment_cost=(
            float(result["assignment_cost"])
            if result.get("assignment_cost") is not None else None
        ),
        optimality_gap_percent=gap,
        assignment_objective_cost=(
            float(objective_cost) if objective_cost is not None else None
        ),
        completion_rate=comp_rate,
        assignment_rate=assign_rate,
        physical_completion_rate=phys_rate,
        conflict_count=int(result.get("conflict_count", result.get("physical_conflicts", 0))),
        messages_sent=int(result.get("messages_sent", 0)),
        runtime_ms=float(result.get("runtime_ms", 0.0)),
        assigned_count=int(result.get("assigned_count", round(assign_rate * 10))),
        physically_completed_count=int(result.get("physically_completed_count", round(phys_rate * 10) if phys_rate is not None else 0)),
        unassigned_count=int(result.get("unassigned_count", len(result.get("unassigned_tasks", [])))),
        missed_deadline_count=int(result.get("missed_deadlines_count", 0)),
        travel_distance=(
            float(result.get("travel_distance", result.get("total_travel_distance", 0.0)))
            if result.get("travel_distance", result.get("total_travel_distance", 0.0)) is not None
            else None
        ),
        training_time_s=training_time_s,
        physical_execution_cost=(
            float(result["physical_execution_cost"])
            if result.get("physical_execution_cost") is not None else None
        ),
        allocation_conflict_penalty=float(result.get("allocation_conflict_penalty", 0.0)),
        allocation_runtime_ms=(
            float(result["allocation_runtime_ms"])
            if result.get("allocation_runtime_ms") is not None else None
        ),
        physical_simulation_runtime_ms=(
            float(result["physical_simulation_runtime_ms"])
            if result.get("physical_simulation_runtime_ms") is not None else None
        ),
        allocation_conflict_count=int(result.get("allocation_conflict_count", 0)),
        physical_conflict_count=int(result.get("physical_conflict_count", 0)),
        allocation_unassigned_count=(
            int(result["allocation_unassigned_count"])
            if result.get("allocation_unassigned_count") is not None else None
        ),
        physical_unassigned_count=(
            int(result["physical_unassigned_count"])
            if result.get("physical_unassigned_count") is not None else None
        ),
        static_assignment_cost=(
            float(result["static_assignment_cost"])
            if result.get("static_assignment_cost") is not None else None
        ),
        seq_opt_reference_cost=seq_opt_reference_cost,
        e2e_gap_vs_seq_opt_percent=e2e_gap
    )
