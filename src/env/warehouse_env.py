import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
from src.env.entities import Location, Task, Robot
from src.communication.channel import CommunicationChannel
from config import DEFAULT_CONFIG, SimulationConfig

class DiscreteSpace:
    def __init__(self, n: int):
        self.n = n
        self.shape = ()
        self.dtype = np.int64

    def sample(self) -> int:
        return int(np.random.randint(0, self.n))

    def contains(self, x) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n

    def __repr__(self):
        return f"Discrete({self.n})"

class BoxSpace:
    def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x) -> bool:
        return isinstance(x, np.ndarray) and x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def __repr__(self):
        return f"Box({self.shape}, dtype={self.dtype})"


class WarehouseMRTAEnv:
    """
    Custom Gym-style Multi-Robot Task Allocation Simulation Environment.
    Faithful to original Ideation Report:
    - Maintains separate local task beliefs for every robot.
    - Zero global state leakage: observation vectors and action masks use only locally
      known information and received communication channel messages.
    - Communication channel integration (L0 to L4) governing message arrival.
    - Realistic multi-step physical execution: pickup travel, delivery travel, handling dwell.
    - Common physical execution support for evaluating both static allocators and MARL.
    """
    def __init__(self, comm_channel: Optional[CommunicationChannel] = None, config: SimulationConfig = DEFAULT_CONFIG):
        self.config = config
        self.channel = comm_channel or CommunicationChannel("L2_Medium", config)
        self.robots: List[Robot] = []
        self.tasks: List[Task] = []
        self.current_step: int = 0
        self.rng = np.random.RandomState(42)
        # 'original': r = 20*completed - travel - 5*conflicts - 0.1*messages (original IDQN checkpoint)
        # 'aligned' : r = 50*on-time completed - travel - 10*conflicts - 0.1*messages, i.e. the cost terms
        #             of the evaluation (P_u = 50, travel, P_c = 10); same form as proposal Eq. 12/18.
        self.reward_mode = 'original'

        # Active physical execution state for each robot:
        # dict(phase="to_pickup"|"to_delivery"|"handling", task=Task, dwell_remaining=int)
        self.robot_active_state: Dict[int, Optional[Dict[str, Any]]] = {}

        # Decentralized local beliefs for each robot:
        # Tasks robot knows are claimed by teammates (received via channel)
        self.local_known_claimed: Dict[int, Set[int]] = {}
        # Tasks completed by this robot
        self.local_completed: Dict[int, Set[int]] = {}

        # Observation dimension:
        # 7 ego-features + M tasks * 8 features = 7 + 10 * 8 = 87
        self.obs_dim = 7 + self.config.n_tasks * 8
        self.single_observation_space = BoxSpace(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.single_action_space = DiscreteSpace(self.config.n_tasks + 1)  # 0..M-1 tasks, index M is WAIT

        self.observation_space = {i: self.single_observation_space for i in range(self.config.n_robots)}
        self.action_space = {i: self.single_action_space for i in range(self.config.n_robots)}

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_step = 0
        self.channel.reset()

        # Instantiate 3 robots with fixed capability distribution
        self.robots = []
        for i in range(self.config.n_robots):
            pos = Location(
                x=int(self.rng.randint(0, self.config.grid_width)),
                y=int(self.rng.randint(0, self.config.grid_height))
            )
            # Robot 0: {"A"}, Robot 1: {"B"}, Robot 2: {"A", "B"}
            cap = {"A"} if i == 0 else ({"B"} if i == 1 else {"A", "B"})
            self.robots.append(Robot(
                id=i,
                position=pos,
                capabilities=cap,
                battery=float(self.config.robot_battery_max),
                capacity_limit=int(self.config.robot_capacity),
                tasks_assigned=[],
                current_task_id=None,
                busy_until_step=0,
                total_travel_distance=0
            ))

        # Instantiate 10 tasks with balanced capability requirements and deadlines
        self.tasks = []
        for j in range(self.config.n_tasks):
            pickup = Location(
                x=int(self.rng.randint(0, self.config.grid_width)),
                y=int(self.rng.randint(0, self.config.grid_height))
            )
            delivery = Location(
                x=int(self.rng.randint(0, self.config.grid_width)),
                y=int(self.rng.randint(0, self.config.grid_height))
            )
            req = "A" if (j % 2 == 0) else "B"
            duration = int(self.rng.randint(self.config.task_duration_min, self.config.task_duration_max + 1))
            deadline = int(self.rng.randint(self.config.deadline_min, self.config.deadline_max + 1))
            self.tasks.append(Task(
                id=j,
                pickup=pickup,
                delivery=delivery,
                capability_req=req,
                release_time=0,
                duration=duration,
                deadline=deadline,
                assigned_robot=None,
                completed=False,
                completion_time=None
            ))

        self.robot_active_state = {r.id: None for r in self.robots}
        self.local_known_claimed = {r.id: set() for r in self.robots}
        self.local_completed = {r.id: set() for r in self.robots}

        obs = self._get_observations()
        masks = self._get_action_masks()

        info: Dict[str, Any] = {"action_masks": masks}
        for r_id, m in masks.items():
            info[r_id] = m

        return obs, info

    def _process_incoming_messages(self):
        """
        Processes delivered messages from each robot's inbox to update local beliefs.
        At L0, no messages are ever received.
        """
        for r in self.robots:
            inbox = self.channel.receive_messages(r.id)
            for msg in inbox:
                if msg.msg_type == "CLAIM":
                    t_id = msg.payload.get("task_id")
                    if t_id is not None:
                        self.local_known_claimed[r.id].add(t_id)

    def _get_action_masks(self) -> Dict[int, np.ndarray]:
        """
        Computes boolean action masks for each robot using ONLY local beliefs.
        Zero global state leakage: does NOT check task.assigned_robot directly.
        """
        masks = {}
        M = self.config.n_tasks

        for r in self.robots:
            mask = np.zeros(M + 1, dtype=bool)
            mask[M] = True  # WAIT action is always valid

            is_busy = (r.current_task_id is not None) or (self.current_step < r.busy_until_step)
            has_capacity = len(r.tasks_assigned) < r.capacity_limit
            has_battery = r.battery > 5.0

            if not is_busy and has_capacity and has_battery:
                for j, task in enumerate(self.tasks):
                    # Local belief check: not locally known claimed, not locally completed, not currently held
                    not_claimed_by_other = task.id not in self.local_known_claimed[r.id]
                    not_completed_by_me = task.id not in self.local_completed[r.id]
                    not_held_by_me = task.id not in r.tasks_assigned

                    if not_claimed_by_other and not_completed_by_me and not_held_by_me:
                        # Capability check
                        if task.capability_req in r.capabilities:
                            # Deadline feasibility check
                            est_finish = r.estimated_completion_time(task, self.current_step)
                            if est_finish <= task.deadline:
                                # Battery check
                                dist = r.position.manhattan_distance(task.pickup) + task.travel_distance
                                bat_req = dist * self.config.battery_cost_per_unit_dist
                                if r.battery >= bat_req:
                                    mask[j] = True

            masks[r.id] = mask
        return masks

    def _get_observations(self) -> Dict[int, np.ndarray]:
        """
        Generates normalized 87-dimensional observation vectors for each robot.
        Vector layout:
        Ego-robot features (7):
          [0]: normalized x position
          [1]: normalized y position
          [2]: normalized battery
          [3]: remaining capacity fraction
          [4]: busy flag
          [5]: capability A flag
          [6]: capability B flag
        For each of M tasks (8 features each):
          [7 + 8*j + 0]: pickup x / W
          [7 + 8*j + 1]: pickup y / H
          [7 + 8*j + 2]: delivery x / W
          [7 + 8*j + 3]: delivery y / H
          [7 + 8*j + 4]: remaining time to deadline / max_steps
          [7 + 8*j + 5]: estimated assignment cost / max_possible_cost
          [7 + 8*j + 6]: teammate claim received (1.0 if received via channel, else 0.0)
          [7 + 8*j + 7]: local availability belief (1.0 if believed available, else 0.0)
        """
        obs = {}
        W = float(self.config.grid_width)
        H = float(self.config.grid_height)
        max_dist = W + H + W + H

        for r in self.robots:
            is_busy = 1.0 if (r.current_task_id is not None or self.current_step < r.busy_until_step) else 0.0
            r_feat = [
                r.position.x / W,
                r.position.y / H,
                r.battery / float(self.config.robot_battery_max),
                float(r.capacity_limit - len(r.tasks_assigned)) / float(r.capacity_limit),
                is_busy,
                1.0 if "A" in r.capabilities else 0.0,
                1.0 if "B" in r.capabilities else 0.0
            ]

            t_feats = []
            for t in self.tasks:
                time_left = max(0, t.deadline - self.current_step) / float(self.config.max_steps)
                est_cost = r.estimated_cost_to_task(t, self.config.alpha, self.config.beta)
                norm_cost = min(1.0, est_cost / float(max_dist))

                # Information received from teammates via channel
                teammate_claimed = 1.0 if (t.id in self.local_known_claimed[r.id]) else 0.0

                # Local belief availability: not claimed by teammate, not completed, deadline valid
                local_avail = 1.0 if (
                    t.id not in self.local_known_claimed[r.id]
                    and t.id not in self.local_completed[r.id]
                    and self.current_step <= t.deadline
                ) else 0.0

                t_feats.extend([
                    t.pickup.x / W,
                    t.pickup.y / H,
                    t.delivery.x / W,
                    t.delivery.y / H,
                    time_left,
                    norm_cost,
                    teammate_claimed,
                    local_avail
                ])

            obs[r.id] = np.array(r_feat + t_feats, dtype=np.float32)
        return obs

    def step(
        self,
        actions: Dict[int, int],
        emit_claim_messages: bool = True
    ) -> Tuple[Dict[int, np.ndarray], float, bool, Dict[str, Any]]:
        messages_before = self.channel.total_messages
        self.current_step += 1

        # 1. Flush buffered messages if this is a synchronization step
        self.channel.flush_buffer_if_period(self.current_step)

        # 2. Update local beliefs from received messages
        self._process_incoming_messages()

        M = self.config.n_tasks
        conflicts = 0
        step_travel_dist = 0
        tasks_completed_this_step = 0
        ontime_this_step = 0

        # 3. Process task claims from idle robots
        step_selections: Dict[int, List[int]] = {}
        for r_id, action in actions.items():
            robot = self.robots[r_id]
            is_busy = (robot.current_task_id is not None) or (self.current_step < robot.busy_until_step)

            if not is_busy and action < M:
                task_id = action
                if task_id not in step_selections:
                    step_selections[task_id] = []
                step_selections[task_id].append(r_id)

                # Broadcast claim to teammates (subject to channel timing & budget)
                est_cost = robot.estimated_cost_to_task(self.tasks[task_id], self.config.alpha, self.config.beta)
                if emit_claim_messages:
                    self.channel.broadcast(
                        sender_id=r_id,
                        msg_type="CLAIM",
                        payload={"task_id": task_id, "cost": est_cost},
                        current_step=self.current_step
                    )

        # Resolve simultaneous selections at this step
        for task_id, contenders in step_selections.items():
            task = self.tasks[task_id]

            if len(contenders) > 1:
                # Concurrent collision!
                conflicts += (len(contenders) - 1)
                # Winner selection: lower estimated cost, tie-break lower robot ID
                contenders.sort(key=lambda rid: (
                    self.robots[rid].estimated_cost_to_task(task, self.config.alpha, self.config.beta),
                    rid
                ))

            winner_id = contenders[0]

            # Physical assignment check: can only be assigned if not already owned by another robot
            if task.assigned_robot is None:
                task.assigned_robot = winner_id
                winner_robot = self.robots[winner_id]
                winner_robot.tasks_assigned.append(task_id)
                winner_robot.current_task_id = task_id

                # Start physical execution: Phase 1 (travel to pickup)
                self.robot_active_state[winner_id] = {
                    "phase": "to_pickup",
                    "task": task,
                    "dwell_remaining": task.duration
                }
            else:
                # Task was already claimed in a previous step by another robot
                # This robot suffered an uncoordinated conflict due to communication lag/L0!
                conflicts += 1

        # 4. Realistic multi-step physical execution for active robots
        for r in self.robots:
            state = self.robot_active_state[r.id]
            if state is not None:
                task = state["task"]

                if state["phase"] == "to_pickup":
                    # Move 1 grid step towards pickup location
                    if r.position != task.pickup:
                        energy = self.config.battery_cost_per_unit_dist
                        if r.battery + 1e-9 < energy:
                            # Battery depletion stops physical movement. The task
                            # remains incomplete and is counted as a missed deadline.
                            continue
                        r.move_towards(task.pickup)
                        r.battery = max(0.0, r.battery - energy)
                        step_travel_dist += 1
                        r.total_travel_distance += 1

                    if r.position == task.pickup:
                        # Reached pickup; transition to Phase 2 (travel to delivery)
                        state["phase"] = "to_delivery"

                elif state["phase"] == "to_delivery":
                    # Move 1 grid step towards delivery location
                    if r.position != task.delivery:
                        energy = self.config.battery_cost_per_unit_dist
                        if r.battery + 1e-9 < energy:
                            # Battery depletion stops physical movement.
                            continue
                        r.move_towards(task.delivery)
                        r.battery = max(0.0, r.battery - energy)
                        step_travel_dist += 1
                        r.total_travel_distance += 1

                    if r.position == task.delivery:
                        # Reached delivery; transition to Phase 3 (service handling dwell)
                        state["phase"] = "handling"

                elif state["phase"] == "handling":
                    # Dwell at delivery location for handling duration
                    state["dwell_remaining"] -= 1
                    if state["dwell_remaining"] <= 0:
                        # Phase 4: Task Completion!
                        task.completed = True
                        task.completion_time = self.current_step
                        tasks_completed_this_step += 1
                        if self.current_step <= task.deadline:
                            ontime_this_step += 1

                        self.local_completed[r.id].add(task.id)
                        r.current_task_id = None
                        self.robot_active_state[r.id] = None

        # 5. Shared team reward formulation (Ideation Report Eq. 12 & 18)
        new_messages = self.channel.total_messages - messages_before
        reward = (
            20.0 * tasks_completed_this_step
            - 1.0 * step_travel_dist
            - 5.0 * conflicts
            - 0.1 * new_messages
        )

        if self.reward_mode == 'aligned':
            reward = 50.0 * ontime_this_step - 1.0 * step_travel_dist - 10.0 * conflicts - 0.1 * new_messages

        # 6. Check episode termination
        all_completed = all(t.completed for t in self.tasks)
        time_limit = self.current_step >= self.config.max_steps
        all_deadlines_passed = all(t.completed or self.current_step > t.deadline for t in self.tasks)
        done = all_completed or time_limit or all_deadlines_passed

        next_obs = self._get_observations()
        next_masks = self._get_action_masks()

        info = {
            "step": self.current_step,
            "tasks_completed_this_step": tasks_completed_this_step,
            "total_tasks_completed": sum(1 for t in self.tasks if t.completed),
            "step_travel_dist": step_travel_dist,
            "conflicts_this_step": conflicts,
            "messages_sent_this_step": new_messages,
            "action_masks": next_masks
        }
        for r_id, m in next_masks.items():
            info[r_id] = m

        return next_obs, reward, done, info

    def run_physical_execution(self, assignments: Dict[int, List[int]]) -> Dict[str, Any]:
        """
        Executes static allocation assignments in this common physical simulation environment.
        Provides fair, apples-to-apples sequential physical execution metrics for all methods.
        """
        task_queues = {r.id: list(assignments.get(r.id, [])) for r in self.robots}
        M = self.config.n_tasks
        total_conflicts = 0

        # A static allocation must contain each task at most once and only known task IDs.
        flattened = [task_id for queue in task_queues.values() for task_id in queue]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Static allocation assigns the same task to multiple robots.")
        valid_task_ids = {task.id for task in self.tasks}
        unknown = set(flattened) - valid_task_ids
        if unknown:
            raise ValueError(f"Static allocation contains unknown task IDs: {sorted(unknown)}")

        while self.current_step < self.config.max_steps:
            actions = {}
            for r in self.robots:
                is_busy = (r.current_task_id is not None) or (self.current_step < r.busy_until_step)
                if (
                    not is_busy
                    and len(task_queues[r.id]) > 0
                    and r.battery + 1e-9 >= self.config.battery_cost_per_unit_dist
                ):
                    actions[r.id] = task_queues[r.id].pop(0)
                else:
                    actions[r.id] = M  # WAIT

            # Allocation is already fixed; do not emit extra CLAIM messages during
            # physical replay. This keeps communication overhead attributable to
            # the allocator rather than the shared execution simulator.
            obs, reward, done, info = self.step(actions, emit_claim_messages=False)
            total_conflicts += info.get("conflicts_this_step", 0)

            # Check if all robots finished their assigned queues and are idle
            all_queues_empty = all(len(q) == 0 for q in task_queues.values())
            all_robots_idle = all(r.current_task_id is None for r in self.robots)
            if all_queues_empty and all_robots_idle:
                break

            if done:
                break

        # Calculate final physical execution metrics
        physically_completed = [t.id for t in self.tasks if t.completed and (t.completion_time is not None and t.completion_time <= t.deadline)]
        missed_deadlines = [
            t.id for t in self.tasks
            if t.assigned_robot is not None
            and (
                not t.completed
                or t.completion_time is None
                or t.completion_time > t.deadline
            )
        ]
        unassigned = [t.id for t in self.tasks if t.assigned_robot is None]

        total_travel = sum(r.total_travel_distance for r in self.robots)
        unassigned_penalty = len(unassigned) * self.config.penalty_unassigned
        missed_deadline_penalty = len(missed_deadlines) * self.config.penalty_unassigned
        conflict_penalty = total_conflicts * self.config.penalty_conflict
        total_cost = total_travel + unassigned_penalty + missed_deadline_penalty + conflict_penalty

        return {
            "physically_completed_tasks": physically_completed,
            "physically_completed_count": len(physically_completed),
            "physical_completion_rate": len(physically_completed) / float(M) if M > 0 else 0.0,
            "missed_deadlines_count": len(missed_deadlines),
            "unassigned_count": len(unassigned),
            "total_travel_distance": total_travel,
            "total_physical_cost": total_cost,
            "physical_conflicts": total_conflicts,
            "messages_sent": self.channel.total_messages
        }
