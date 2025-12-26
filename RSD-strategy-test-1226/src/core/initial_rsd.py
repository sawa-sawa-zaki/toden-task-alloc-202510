
"""
Phase Aのみ（直前調整なし）を、元プロジェクトのロジックに合わせて実装する。

本実験では、期待値を「各期の優先順序の列」で厳密に取れる（A=2 なら 2^T 通り）ため、
Monte Carlo ではなく **全順序列の列挙**でEUを計算する関数も提供する。

割当は alloc[a][m][t]（0/1）で表す。
効用は Σ_{m,t} alloc[a][m][t] * true_vals_a[m][t] （元solver.py準拠）
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

def run_phaseA_given_orders(cfg, report_vals: np.ndarray, order_seq: List[List[int]]) -> np.ndarray:
    """
    Phase A（量モデル・修正版）

    alloc[a,m,t] は割当量（int >= 0）
    """
    alloc = np.zeros((cfg.A, cfg.M, cfg.T), dtype=int)
    remaining = list(cfg.Q)

    for t in range(cfg.T):
        priority_order = order_seq[t]
        window_end = min(cfg.T, t + cfg.WINDOW)

        for agent in priority_order:
            # 残需要がある限り 1単位ずつ割当
            while remaining[agent] > 0:
                best_slot = None
                best_val = -float("inf")

                for w_t in range(t, window_end):
                    safe_cap = max(
                        0,
                        int(cfg.BASE_SUPPLY[w_t] - cfg.BUFFER[w_t])
                    )

                    # 時間帯の総量制約
                    if alloc[:, :, w_t].sum() >= safe_cap:
                        continue

                    for m in range(cfg.M):
                        # 機械容量制約
                        if alloc[:, m, w_t].sum() >= cfg.MACHINE_CAPACITY:
                            continue

                        val = report_vals[agent, m, w_t]
                        if val > best_val:
                            best_val = val
                            best_slot = (m, w_t)

                # outside option
                if best_slot is None or best_val <= 0:
                    break

                m, w_t = best_slot
                alloc[agent, m, w_t] += 1
                remaining[agent] -= 1

    return alloc


def expected_utility_exact_orders(cfg, report_vals: np.ndarray, true_vals_a: np.ndarray, true_vals_b: np.ndarray):
    """
    A=2を想定。各期の優先順序が独立に一様 [0,1] / [1,0] を取るときの期待効用を厳密計算。
    2^T 通りの order_seq を列挙して平均。
    """
    assert cfg.A == 2, "expected_utility_exact_orders は A=2 を想定しています"
    orders = [[0,1],[1,0]]

    total_u = np.zeros(cfg.A, dtype=float)
    count = 0
    for choice in np.ndindex(*(2 for _ in range(cfg.T))):
        order_seq = [orders[c] for c in choice]
        alloc = run_phaseA_given_orders(cfg, report_vals, order_seq)
        u0 = float(np.sum(alloc[0] * true_vals_a))
        u1 = float(np.sum(alloc[1] * true_vals_b))
        total_u[0] += u0
        total_u[1] += u1
        count += 1
    return total_u / max(count, 1)
