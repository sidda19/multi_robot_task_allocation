"""Train the shared-parameter MARL policy (about 4 minutes on a CPU).
Run from the project root:  python scripts/train_shared_marl.py   -> models/shared_dqfd/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.marl.shared_dqfd import train_shared_dqfd

if __name__ == "__main__":
    res = train_shared_dqfd()
    print(res["metadata"])
