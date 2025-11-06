import json
import numpy as np
from typing import Tuple

def load_linear_values_json(
    path: str,
    num_agents: int,
    num_machines: int,
    time_horizon: int,
    allow_ties: bool = True,
) -> np.ndarray:
    """
    JSONスキーマ:
    {
      "num_agents": A,
      "num_machines": M,
      "time_horizon": T,
      "agents": [
        { "id": int, "values": [[m0_t0, m0_t1, ...], [m1_t0, m1_t1, ...], ...] },
        ...
      ]
    }

    - サイズが不足する場合は 0.0 で埋める
    - 余剰分は切り捨て
    - allow_ties=True の場合でも、値はそのまま（丸めはしない）
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vals = np.zeros((num_agents, num_machines, time_horizon), dtype=float)

    agents = data.get("agents", [])
    for agent in agents:
        a = int(agent.get("id", -1))
        if not (0 <= a < num_agents):
            continue
        mat = agent.get("values", [])
        # mat は [machine][timeslot]
        for m in range(min(num_machines, len(mat))):
            row = mat[m]
            for t in range(min(time_horizon, len(row))):
                try:
                    vals[a, m, t] = float(row[t])
                except Exception:
                    vals[a, m, t] = 0.0

    return vals
