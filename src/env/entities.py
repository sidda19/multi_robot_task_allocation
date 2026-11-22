from dataclasses import dataclass, field
from typing import Set, Optional, List

@dataclass
class Location:
    x: int
    y: int

    def manhattan_distance(self, other: "Location") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def to_tuple(self):
        return (self.x, self.y)

@dataclass
class Task:
    id: int
    pickup: Location
    delivery: Location
    capability_req: str
    release_time: int = 0
    duration: int = 2
    deadline: int = 30
    assigned_robot: Optional[int] = None
    completed: bool = False
    completion_time: Optional[int] = None

    @property
    def travel_distance(self) -> int:
        return self.pickup.manhattan_distance(self.delivery)

@dataclass
class Robot:
    id: int
    position: Location
    capabilities: Set[str]
    battery: float = 100.0
    capacity_limit: int = 3
    tasks_assigned: List[int] = field(default_factory=list)
    current_task_id: Optional[int] = None
    busy_until_step: int = 0
    total_travel_distance: int = 0

    def can_perform(self, task: Task) -> bool:
        return (task.capability_req in self.capabilities and 
                len(self.tasks_assigned) < self.capacity_limit and 
                self.battery > 10.0)

    def estimated_cost_to_task(self, task: Task, alpha: float = 1.0, beta: float = 0.5) -> float:
        dist_to_pickup = self.position.manhattan_distance(task.pickup)
        task_dist = task.travel_distance
        total_dist = dist_to_pickup + task_dist
        resource_cost = total_dist * 0.8 + task.duration * 0.5
        return alpha * total_dist + beta * resource_cost

    def estimated_completion_time(self, task: Task, current_step: int) -> int:
        dist_to_pickup = self.position.manhattan_distance(task.pickup)
        task_dist = task.travel_distance
        start_step = max(current_step, self.busy_until_step)
        return start_step + dist_to_pickup + task_dist + task.duration

    def move_towards(self, target: Location):
        import numpy as np
        dx = np.sign(target.x - self.position.x)
        dy = np.sign(target.y - self.position.y)
        if dx != 0:
            self.position = Location(self.position.x + int(dx), self.position.y)
        elif dy != 0:
            self.position = Location(self.position.x, self.position.y + int(dy))
