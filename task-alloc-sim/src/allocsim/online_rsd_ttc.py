import numpy as np
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from allocsim.julia_bridge import (
    call_julia_trial, linearize_levels,
    JuliaNotFound, JuliaFailed
)

# ==============================================================
# Config structure
# ==============================================================

@dataclass
class Config:
    A: int               # players
    M: int               # machines
    T: int               # time periods
    q: List[int]         # demands per player
    barS: List[int]      # expected supply per t
    buffer: List[int]    # buffer per t
    window: int
    shock_p: float       # probability of +/- shock


# ==============================================================
# Core Mechanics
# ==============================================================

def rsd_step(cfg: Config, reports, alloc, residual, rsd_order, t0: int):
    """
    reports[i] は「レベル構造」の報告を想定：
      reports[i] == [ [(m,t1), (m,t1), ...],  [(m,t1), ...],  ... ]
    'OUT' は含まれていても無視する（文字列はスキップ）。
    """
    def iter_cells(report_i):
        for level in report_i:
            if isinstance(level, list):
                for cell in level:
                    if isinstance(cell, (list, tuple)) and len(cell) == 2:
                        m, t = int(cell[0]), int(cell[1])
                        yield m, t
            # 文字列（例: 'OUT'）は無視

    for i in rsd_order:
        report_i = reports[i]
        for (m, t) in iter_cells(report_i):
            if t < 0 or t >= cfg.T:
                continue
            if residual[t] <= 0:
                continue
            if alloc[i][m][t] == 0:
                alloc[i][m][t] = 1
                residual[t] -= 1
                # 1本だけ確保したら、そのプレイヤーのRSDは終了（需要=1前提）
                break



def ttc_adjust(cfg: Config, values, alloc, residual, shocks):
    """Simplified TTC: adjust allocations after shock."""
    A, M, T = cfg.A, cfg.M, cfg.T
    for t in range(T):
        realized = cfg.barS[t] + shocks[t]
        assigned = sum(alloc[i][m][t] for i in range(A) for m in range(M))
        diff = realized - assigned
        if diff > 0:
            # promote best unassigned
            for _ in range(diff):
                best_i, best_m, best_v = None, None, -1e9
                for i in range(A):
                    assigned_t = any(alloc[i][m][t] == 1 for m in range(M))
                    if not assigned_t:
                        for m in range(M):
                            v = values[i][m][t]
                            if v > best_v:
                                best_i, best_m, best_v = i, m, v
                if best_i is not None:
                    alloc[best_i][best_m][t] = 1
        elif diff < 0:
            # remove lowest value
            for _ in range(-diff):
                worst_i, worst_m, worst_v = None, None, 1e9
                for i in range(A):
                    for m in range(M):
                        if alloc[i][m][t] == 1:
                            v = values[i][m][t]
                            if v < worst_v:
                                worst_i, worst_m, worst_v = i, m, v
                if worst_i is not None:
                    alloc[worst_i][worst_m][t] = 0


def compute_utility(values, alloc):
    """Return each player's utility."""
    A, M, T = len(values), len(values[0]), len(values[0][0])
    u = np.zeros(A)
    for i in range(A):
        for m in range(M):
            for t in range(T):
                if alloc[i][m][t] == 1:
                    u[i] += values[i][m][t]
    return u


def run_online_rsd_ttc(cfg: Config, values, reports, seed=None):
    """Simulate online RSD→TTC."""
    np.random.seed(seed)
    random.seed(seed)
    A, M, T = cfg.A, cfg.M, cfg.T
    alloc = np.zeros((A, M, T), dtype=int)
    residual = np.array(cfg.barS, dtype=int)
    rsd_order = list(range(A))
    random.shuffle(rsd_order)

    # RSD allocation phase
    for t in range(T):
        rsd_step(cfg, reports, alloc, residual, rsd_order, t)

    # Random supply shocks
    shocks = []
    for _ in range(T):
        r = random.random()
        if r < cfg.shock_p / 2:
            shocks.append(-1)
        elif r < cfg.shock_p:
            shocks.append(1)
        else:
            shocks.append(0)

    # TTC adjustment
    ttc_adjust(cfg, values, alloc, residual, shocks)
    return {"alloc": alloc, "residual": residual, "shocks": shocks}


# ==============================================================
# Evaluate ε-IC
# ==============================================================

def evaluate_regret_once(cfg: Config, values, reports_truth, reports_dev, seed=0, use_julia=False):
    rng = np.random.RandomState(seed)
    A = cfg.A
    rsd_order = list(rng.permutation(A) + 1)  # Juliaは1-based
    shocks = []
    for _ in range(cfg.T):
        r = rng.rand()
        if r < cfg.shock_p / 2:
            shocks.append(-1)
        elif r < cfg.shock_p:
            shocks.append(1)
        else:
            shocks.append(0)

    if use_julia:
        try:
            reports_linear = [linearize_levels(r) for r in reports_truth]
            out_truth = call_julia_trial(cfg, values, reports_linear, rsd_order, shocks)
            u_truth = np.array(out_truth["utility"], dtype=float)

            reports_linear_dev = [linearize_levels(r) for r in reports_dev]
            out_dev = call_julia_trial(cfg, values, reports_linear_dev, rsd_order, shocks)
            u_dev = np.array(out_dev["utility"], dtype=float)

            return u_truth, u_dev
        except (JuliaNotFound, JuliaFailed) as e:
            print(f"[WARN] Julia failed: {e}. Falling back to Python.")
            pass

    # fallback Python path
    out_truth = run_online_rsd_ttc(cfg, values, reports_truth, seed=seed)
    out_dev = run_online_rsd_ttc(cfg, values, reports_dev, seed=seed)
    u_truth = compute_utility(values, out_truth["alloc"])
    u_dev = compute_utility(values, out_dev["alloc"])
    return u_truth, u_dev


def evaluate_ic(cfg: Config, values, truth, devs, samples=300, seed0=0, use_julia=False):
    """Monte Carlo over profiles"""
    A = cfg.A
    regret_means = []
    results = []
    for k, (reports_truth, reports_dev) in enumerate(zip(truth, devs)):
        regrets = []
        for s in range(samples):
            u_truth, u_dev = evaluate_regret_once(cfg, values, reports_truth, reports_dev,
                                                  seed=seed0 + s, use_julia=use_julia)
            regrets.append(u_dev - u_truth)
        regrets = np.array(regrets)
        regret_mean = regrets.mean(axis=0)
        regret_max = regrets.max(axis=0)
        results.append((k, regret_mean, regret_max))
        regret_means.append(np.mean(regret_mean))
    return {
        "regret_mean_avg": np.mean(regret_means),
        "regret_max": np.max([r[2].max() for r in results]),
        "results": results
    }
