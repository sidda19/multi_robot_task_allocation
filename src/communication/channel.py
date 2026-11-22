from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from config import DEFAULT_CONFIG

@dataclass
class Message:
    sender_id: int
    msg_type: str  # 'CLAIM', 'BID', 'STATUS', 'RELEASE'
    payload: Dict[str, Any]
    step_created: int
    delivered: bool = False

class CommunicationChannel:
    """
    Simulates inter-robot communication under five discrete levels (L0 to L4):
    L0 (None):    k = inf, B = 0     -> No communication permitted. Zero exchange.
    L1 (Low):     k = 10,  B = 6     -> Infrequent sync every 10 steps, 6 msgs/robot budget.
    L2 (Medium):  k = 5,   B = 15    -> Regular sync every 5 steps, 15 msgs/robot budget.
    L3 (High):    k = 1,   B = 100   -> Per-step sync every 1 step, 100 msgs/robot budget.
    L4 (Full):    k = 0,   B = inf   -> Instantaneous broadcast, unbounded messages.
    """
    def __init__(self, comm_level_name: str = "L2_Medium", config=DEFAULT_CONFIG):
        self.config = config
        self.level_name = comm_level_name
        comm_params = config.COMM_LEVELS.get(comm_level_name, config.COMM_LEVELS["L2_Medium"])
        self.k = comm_params["k"]
        self.budget_per_robot = comm_params["budget"]
        self.description = comm_params["description"]

        self.messages_sent_by_robot: Dict[int, int] = {i: 0 for i in range(config.n_robots)}
        self.total_messages: int = 0
        self.inbox: Dict[int, List[Message]] = {i: [] for i in range(config.n_robots)}
        self.outbox_buffer: List[Message] = []
        # The auctioneer is a logical coordinator, not one of the physical robots.
        self.coordinator_inbox: List[Message] = []
        self.coordinator_buffer: List[Message] = []

    def reset(self):
        self.messages_sent_by_robot = {i: 0 for i in range(self.config.n_robots)}
        self.total_messages = 0
        self.inbox = {i: [] for i in range(self.config.n_robots)}
        self.outbox_buffer = []
        self.coordinator_inbox = []
        self.coordinator_buffer = []

    def can_communicate_at_step(self, step: int) -> bool:
        """
        Determines whether communication is active at the given discrete step.
        """
        if self.k == float("inf"):
            return False  # L0: Never communicates
        if self.k == 0:
            return True   # L4: Always communicates (instantaneous)
        return (step % self.k) == 0

    def broadcast(self, sender_id: int, msg_type: str, payload: Dict[str, Any], current_step: int) -> bool:
        """
        Broadcasts a message from sender_id to all other robots.
        Enforces:
        - L0 blockage (k = inf)
        - Per-robot cumulative message budget constraint (m <= B)
        - Latency buffering until sync step (unless k = 0)
        """
        if self.k == float("inf") or self.budget_per_robot <= 0:
            return False  # L0: No communication allowed

        if self.messages_sent_by_robot[sender_id] >= self.budget_per_robot:
            return False  # Message budget exhausted for this sender

        msg = Message(
            sender_id=sender_id,
            msg_type=msg_type,
            payload=payload,
            step_created=current_step
        )
        self.messages_sent_by_robot[sender_id] += 1
        self.total_messages += 1

        # L4 (k = 0) delivers immediately
        if self.k == 0:
            msg.delivered = True
            for r_id in range(self.config.n_robots):
                if r_id != sender_id:
                    self.inbox[r_id].append(msg)
        elif self.can_communicate_at_step(current_step):
            msg.delivered = True
            for r_id in range(self.config.n_robots):
                if r_id != sender_id:
                    self.inbox[r_id].append(msg)
        else:
            # Buffer message in outbox until next synchronization step
            self.outbox_buffer.append(msg)

        return True

    def send_to_coordinator(self, sender_id: int, msg_type: str, payload: Dict[str, Any], current_step: int) -> bool:
        """
        Sends one robot-to-auctioneer message under the same latency and per-robot
        budget rules as peer broadcasts. The coordinator is a logical endpoint,
        so a bid is not duplicated into every robot inbox.
        """
        if self.k == float("inf") or self.budget_per_robot <= 0:
            return False
        if self.messages_sent_by_robot[sender_id] >= self.budget_per_robot:
            return False

        msg = Message(
            sender_id=sender_id,
            msg_type=msg_type,
            payload=payload,
            step_created=current_step
        )
        self.messages_sent_by_robot[sender_id] += 1
        self.total_messages += 1

        if self.k == 0 or self.can_communicate_at_step(current_step):
            msg.delivered = True
            self.coordinator_inbox.append(msg)
        else:
            self.coordinator_buffer.append(msg)
        return True

    def receive_coordinator_messages(self) -> List[Message]:
        """Retrieves all messages delivered to the logical auctioneer."""
        messages = list(self.coordinator_inbox)
        self.coordinator_inbox.clear()
        return messages

    def flush_buffer_if_period(self, current_step: int):
        """
        Delivers all buffered outbox messages if current_step matches the communication period k.
        """
        if self.can_communicate_at_step(current_step):
            if self.outbox_buffer:
                for msg in self.outbox_buffer:
                    msg.delivered = True
                    for r_id in range(self.config.n_robots):
                        if r_id != msg.sender_id:
                            self.inbox[r_id].append(msg)
                self.outbox_buffer.clear()
            if self.coordinator_buffer:
                for msg in self.coordinator_buffer:
                    msg.delivered = True
                    self.coordinator_inbox.append(msg)
                self.coordinator_buffer.clear()

    def receive_messages(self, robot_id: int) -> List[Message]:
        """
        Retrieves and clears all pending delivered messages for robot_id.
        """
        msgs = list(self.inbox[robot_id])
        self.inbox[robot_id].clear()
        return msgs

    def get_remaining_budget(self, robot_id: int) -> float:
        if self.budget_per_robot == float("inf"):
            return float("inf")
        return max(0, self.budget_per_robot - self.messages_sent_by_robot[robot_id])
