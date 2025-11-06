

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2×2（マシン×時間）・プレイヤー2人の完全情報ゲームで、
与えられた選好候補の全組合せについて、偏報（misreport）が
各プレイヤーにとって有利になり得るか（耐戦略性）を総当り検証する。

前提：
- 既存のメカニズム（RSD=シリアル独裁 → TTC調整骨格）
- -1 はそのエージェントに割当禁止（アウトサイド）
- “遅い時間帯を好むことはない”：各マシンごとに v[m, t=2] <= v[m, t=1]（=を許す）
- 単価レベルの候補集合：{-1, 1, 10, 100, 1000}（重複使用可、無差別可）

市場デフォルト（必要なら下の DEFAULT_* を編集）：
- A=2, M=2, T=2
- units_demand = 2
- machine_capacity = [1, 1]（→ 各時刻の合計は最大2）
- supply_forecast_by_t = [2, 2], buffer_by_t = [0, 0], supply_shock_by_t = []
  → 各時刻の実供給ターゲットは 2（電力制約と機械制約が一致）

使い方：
    python scripts/check_strategyproofness.py \
        --seed 123 \
        --out results/strategyproof_2x2.csv
"""

from __future__ import annotations

# --- add this exactly ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # task-alloc-sim/ （プロジェクトルート）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --- end add ---

# 以降は今のまま:
# from src.allocsim.market import Market
# from src.allocsim.process import run_pipeline_for_one_seed
# ...


import os
import csv
import argparse
from types import SimpleNamespace
from itertools import product
import numpy as np

# プロジェクト内モジュール
from src.allocsim.market import Market
from src.allocsim.process import run_pipeline_for_one_seed
from src.allocsim.utils import seed_all

# =========================
# 実験の基本設定（必要なら編集）
# =========================
DEFAULT_A = 2
DEFAULT_M = 2
DEFAULT_T = 2
DEFAULT_UNITS_DEMAND = 2
DEFAULT_MACHINE_CAPACITY = [1, 1]       # per machine
DEFAULT_SUPPLY_FORECAST = [2, 2]        # per time
DEFAULT_BUFFER_BY_T = [0, 0]
DEFAULT_SHOCKS = []                     # [{"t": 0, "delta": 0}, ...]

# 単価レベル集合（-1 は割当禁止）
VALUE_LEVELS = [-1, 1, 10, 100, 1000]


def generate_all_prefs_2x2(levels=VALUE_LEVELS):
    """
    2 machines × 2 times の全選好（v[m,t]）を列挙。
    制約：各 m について v[m,1] >= v[m,2]（t=1 が早い、t=2 が遅いとし、後者を好まない）
          ※ 等号は許す（無差別）
    -1 も選べる（その場合その (m,t) は禁止スロット）
    戻り値：list of np.ndarray shape (M=2, T=2)
    """
    L = levels
    prefs = []
    for a00, a01, a10, a11 in product(L, L, L, L):
        # 形状: [[m0_t1, m0_t2], [m1_t1, m1_t2]] とする（t-indexは 0:早い, 1:遅い）
        v = np.array([[a00, a01],
                      [a10, a11]], dtype=float)

        # 時間単調性: 遅い(t=1) が 早い(t=0) を上回らない
        if v[0, 1] <= v[0, 0] and v[1, 1] <= v[1, 0]:
            prefs.append(v)
    return prefs


def values_to_values_all(agent_values_list):
    """
    agent_values_list: list of np.ndarray, each (M, T)
    returns values_all: np.ndarray (A, M, T)
    """
    return np.stack(agent_values_list, axis=0).astype(float)


def run_mechanism(values_all, rng_seed=0):
    """
    既存パイプライン（RSD→TTC骨格）を1回実行し、配分テンソルを返す。
    values_all: (A, M, T)
    """
    A, M, T = values_all.shape

    market = Market(
        num_agents=A,
        num_machines=M,
        time_horizon=T,
        units_demand=DEFAULT_UNITS_DEMAND,
        supply_forecast_by_t=list(DEFAULT_SUPPLY_FORECAST),
        buffer_by_t=list(DEFAULT_BUFFER_BY_T),
        machine_capacity=list(DEFAULT_MACHINE_CAPACITY),
        supply_shock_by_t=[dict(s) for s in DEFAULT_SHOCKS],
    )
    # cfg はパイプライン内部で参照されない想定だが、一応必要最低限の属性を持たせる
    cfg = SimpleNamespace(
        pipeline=["rsd_initial_global", "ttc_global"],
        market=SimpleNamespace(units_demand=DEFAULT_UNITS_DEMAND),
    )

    seed_all(rng_seed)
    rng = np.random.default_rng(rng_seed)
    result = run_pipeline_for_one_seed(cfg, values_all, market, rng)
    # allocation: (T, A, M)
    return result["allocation"], result


def util_of_agent(values_all, alloc_all, a):
    """
    真の効用で評価： sum_{t,m} alloc[t,a,m] * values[a,m,t]
    values_all: (A,M,T), alloc_all: (T,A,M)
    """
    A, M, T = values_all.shape
    v = values_all[a]          # (M,T)
    alloc = alloc_all[:, a, :] # (T,M)
    # 転置して形合わせ
    return float(np.sum(alloc.T * v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default="results/strategyproof_2x2.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="テストする選好プロファイル数の上限（デバッグ用）")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 候補選好の全列挙
    prefs_all = generate_all_prefs_2x2(VALUE_LEVELS)
    # 2人分のプロファイルを総当り
    profiles = list(product(range(len(prefs_all)), range(len(prefs_all))))

    if args.limit is not None:
        profiles = profiles[:args.limit]

    # CSV 出力準備
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "profile_id",
                "i_true_idx", "j_true_idx",
                "u_i_truth", "u_j_truth",
                "i_profitable", "j_profitable",
                "i_best_gain", "j_best_gain",
                "i_best_mis_idx", "j_best_mis_idx",
                "seed",
            ],
        )
        writer.writeheader()

        # 総当りループ
        for pid, (i_idx, j_idx) in enumerate(profiles):
            Vi_true = prefs_all[i_idx]  # (M,T)
            Vj_true = prefs_all[j_idx]

            values_all_true = values_to_values_all([Vi_true, Vj_true])  # (A=2,M=2,T=2)
            alloc_truth, _ = run_mechanism(values_all_true, rng_seed=args.seed)

            u_i_truth = util_of_agent(values_all_true, alloc_truth, a=0)
            u_j_truth = util_of_agent(values_all_true, alloc_truth, a=1)

            # ---- agent i の偏報探索（j は真実）
            i_best_gain = 0.0
            i_best_mis_idx = -1
            for mis_idx in range(len(prefs_all)):
                Vi_mis = prefs_all[mis_idx]
                values_mis = values_to_values_all([Vi_mis, Vj_true])
                alloc_mis, _ = run_mechanism(values_mis, rng_seed=args.seed)
                # 評価は真の効用
                u_i_mis_true_eval = util_of_agent(values_all_true, alloc_mis, a=0)
                gain = u_i_mis_true_eval - u_i_truth
                if gain > i_best_gain + 1e-12:
                    i_best_gain = gain
                    i_best_mis_idx = mis_idx
            i_profitable = bool(i_best_gain > 1e-12)

            # ---- agent j の偏報探索（i は真実）
            j_best_gain = 0.0
            j_best_mis_idx = -1
            for mis_idx in range(len(prefs_all)):
                Vj_mis = prefs_all[mis_idx]
                values_mis = values_to_values_all([Vi_true, Vj_mis])
                alloc_mis, _ = run_mechanism(values_mis, rng_seed=args.seed)
                u_j_mis_true_eval = util_of_agent(values_all_true, alloc_mis, a=1)
                gain = u_j_mis_true_eval - u_j_truth
                if gain > j_best_gain + 1e-12:
                    j_best_gain = gain
                    j_best_mis_idx = mis_idx
            j_profitable = bool(j_best_gain > 1e-12)

            writer.writerow({
                "profile_id": pid,
                "i_true_idx": i_idx, "j_true_idx": j_idx,
                "u_i_truth": f"{u_i_truth:.6f}",
                "u_j_truth": f"{u_j_truth:.6f}",
                "i_profitable": int(i_profitable),
                "j_profitable": int(j_profitable),
                "i_best_gain": f"{i_best_gain:.6f}",
                "j_best_gain": f"{j_best_gain:.6f}",
                "i_best_mis_idx": i_best_mis_idx,
                "j_best_mis_idx": j_best_mis_idx,
                "seed": args.seed,
            })

    print(f"[done] wrote: {args.out}")


if __name__ == "__main__":
    main()
