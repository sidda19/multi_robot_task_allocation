"""
Reference online policies for the warehouse environment.

They use exactly the same environment, observations-of-teammates (received
messages only), action masks and communication channel as the cooperative MARL
policy, so they are fair yard-sticks for it:

* OnlineGreedyPolicy  - each idle robot takes the cheapest task that its own
  (local) action mask allows. This is the natural *online* Distributed Greedy:
  decisions are spread over the episode, so the synchronisation period k
  really changes what a robot knows when it decides.
* RandomMaskedPolicy  - uniform choice among the actions allowed by the mask
  (including WAIT). A learned policy that cannot beat this has learned nothing
  beyond the action mask.

Both expose `select_actions(obs, masks, training=False)` like
`CooperativeMARLController`, and `bind(env)` is called by the evaluation
runner so the greedy policy can read the robot's own cost estimate.
"""
from typing import Dict

import numpy as np

from config import DEFAULT_CONFIG


class OnlineGreedyPolicy:
    name = "Online Distributed Greedy"

    def __init__(self, config=DEFAULT_CONFIG, tie_break: str = "none", seed: int = 0):
        """tie_break: how robots avoid all choosing the same task when they cannot hear each other.
        "none"       - cheapest valid task (original behaviour; identical robots collide)
        "convention" - a rule agreed in advance and needing NO messages: task j is preferred by
                       robot (j mod N); a robot takes its cheapest valid *preferred* task, else
                       its cheapest valid task
        "random"     - uniform choice among valid tasks whose cost is within 25% of the cheapest
        """
        assert tie_break in ("none", "convention", "random")
        self.config = config
        self.env = None
        self.tie_break = tie_break
        self.rng = np.random.RandomState(seed)

    def bind(self, env) -> None:
        self.env = env

    def select_actions(self, obs, masks: Dict[int, np.ndarray], training: bool = False) -> Dict[int, int]:
        n_tasks = self.config.n_tasks
        actions: Dict[int, int] = {}
        for robot_id, mask in masks.items():
            robot = self.env.robots[robot_id]
            costs = {j: robot.estimated_cost_to_task(self.env.tasks[j], self.config.alpha, self.config.beta)
                     for j in range(n_tasks) if mask[j]}
            if not costs:
                actions[robot_id] = n_tasks                         # WAIT
                continue
            if self.tie_break == "convention":
                n_robots = self.config.n_robots
                own = {j: c for j, c in costs.items() if j % n_robots == robot_id}
                pool = own if own else costs
                actions[robot_id] = min(pool, key=lambda j: (pool[j], j))
            elif self.tie_break == "random":
                best = min(costs.values())
                near = sorted(j for j, c in costs.items() if c <= best * 1.25 + 1e-9)
                actions[robot_id] = int(near[self.rng.randint(len(near))])
            else:
                actions[robot_id] = min(costs, key=lambda j: (costs[j], j))
        return actions


class RandomMaskedPolicy:
    name = "Random Masked Policy"

    def __init__(self, seed: int = 0, config=DEFAULT_CONFIG):
        self.config = config
        self.rng = np.random.RandomState(seed)

    def bind(self, env) -> None:
        pass

    def select_actions(self, obs, masks: Dict[int, np.ndarray], training: bool = False) -> Dict[int, int]:
        actions: Dict[int, int] = {}
        for robot_id, mask in masks.items():
            valid = np.flatnonzero(mask)                           # WAIT is always valid
            actions[robot_id] = int(self.rng.choice(valid))
        return actions
