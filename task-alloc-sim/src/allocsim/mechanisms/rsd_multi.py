import numpy as np
from typing import Dict, List, Tuple


def rsd_multi_round(
    values: np.ndarray,
    capacities_t: np.ndarray,
    remaining_demand: np.ndarray,
    max_per_timeslot: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[int]]:
    """
    マルチユニットRSD（時刻t固定）。
    - values: shape (A, M)  各agentの (machine,t) の単価
    - capacities_t: shape (M,)  時刻tの各machineの残席数
    - remaining_demand: shape (A,) 各agentの残需要（合計）。このtでは min(残需要, max_per_timeslot) まで取得可
    - 戻り値: allocation_t: shape (A, M) in {0,1}, pick_order(list)
    """
    A, M = values.shape
    alloc = np.zeros((A, M), dtype=int)
    per_t_taken = np.zeros(A, dtype=int)

    # ランダム順列
    order = rng.permutation(A).tolist()

    def best_available_machine(a: int) -> int:
        # 残席>0 かつ そのagentが未取得 (max_per_timeslot) の中で単価最大
        best_m, best_v = -1, -1e18
        for m in range(M):
            if capacities_t[m] <= 0:
                continue
            if per_t_taken[a] >= max_per_timeslot:
                continue
            v = values[a, m]
            if v > best_v:
                best_v, best_m = v, m
        return best_m

    progressed = True
    while progressed:
        progressed = False
        for a in order:
            if remaining_demand[a] <= 0:
                continue
            if per_t_taken[a] >= max_per_timeslot:
                continue
            m = best_available_machine(a)
            if m == -1:
                continue
            # 指名
            alloc[a, m] += 1
            capacities_t[m] -= 1
            remaining_demand[a] -= 1
            per_t_taken[a] += 1
            progressed = True

        # 席が尽きる or 全員が上限に達すると停止
        if capacities_t.sum() == 0:
            break

    return alloc, order
