import numpy as np
from typing import Dict

def compute_metrics(values_all: np.ndarray, allocation: np.ndarray, market) -> Dict[str, float]:
    T, A, M = allocation.shape
    welfare = 0.0
    for t in range(T):
        welfare += float((values_all[:, :, t] * allocation[t]).sum())

    total_assigned = int(allocation.sum())
    total_demand = int(A * market.units_demand)
    fill_ratio = (total_assigned / total_demand) if total_demand > 0 else 1.0

    # ★ 供給総数は「変動後」合計で評価
    total_supply = sum(market.total_cap_t_after(t) for t in range(market.time_horizon))
    utilization = (total_assigned / total_supply) if total_supply > 0 else 1.0

    return {
        "welfare": welfare,
        "fill_ratio": fill_ratio,
        "utilization": utilization,
        "assigned": total_assigned,
        "demand": total_demand,
        "supply": total_supply,
    }
