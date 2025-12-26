from __future__ import annotations

"""
実験C（TypeCx: strict order 720）
※ unilateral deviation のみ

- 真の選好：RESTRICTED（216）
- 嘘：TypeCx（厳密順序 720）
- 検証：A 側のみが嘘をつく（B は常に正直）

重要：
- allocation は report で決定
- utility は true type で評価
- outside：
    ・未割当は 0
    ・outside スロットに割当が起きた場合のみ -1
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
from dataclasses import asdict
from tqdm import tqdm

from src.config.base_config import Config, ICConfig
from src.core.generator import (
    generate_consistent_with_truncation,
    ranks_to_vals,
)
from src.core.initial_rsd import run_phaseA_given_orders
from src.utils.results import make_timestamped_dir


# -------------------------------------------------
# 基本関数
# -------------------------------------------------

def expected_alloc_given_reports(cfg: Config, report_vals_pair: np.ndarray) -> np.ndarray:
    """
    固定された report_vals_pair のもとでの PhaseA の期待配分
    （order_seq を 2^T 通り全列挙）
    return: (A,M,T)
    """
    orders = [[0, 1], [1, 0]]
    acc = np.zeros((cfg.A, cfg.M, cfg.T), dtype=float)
    cnt = 0

    for choice in np.ndindex(*(2 for _ in range(cfg.T))):
        order_seq = [orders[c] for c in choice]
        alloc = run_phaseA_given_orders(cfg, report_vals_pair, order_seq)
        acc += alloc
        cnt += 1

    return acc / max(cnt, 1)


def utility(expected_alloc_mt: np.ndarray, true_vals_mt: np.ndarray) -> float:
    """
    ユーザー仕様：
    - 未割当は 0
    - outside スロット割当は true_vals=-1 により減点
    """
    return float(np.sum(expected_alloc_mt * true_vals_mt))


def generate_strict_orders_6():
    """
    6 スロット（M*T=2*3）の厳密順序：720通り
    """
    return list(itertools.permutations(range(6)))


# -------------------------------------------------
# main
# -------------------------------------------------

def run():
    cfg = Config()
    icfg = ICConfig()

    assert cfg.A == 2 and cfg.M == 2 and cfg.T == 3
    assert cfg.STRATEGY_DOMAIN == "RESTRICTED"

    out_dir = make_timestamped_dir(icfg.OUTPUT_BASE_DIR, prefix="expC_TypeCx_unilateral")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------
    # 1) 真の選好（RESTRICTED）
    # ---------------------------------------------
    restricted = generate_consistent_with_truncation(cfg.M, cfg.T)
    true_ranks = [d["ranks"] for d in restricted]
    n_true = len(true_ranks)

    print(f"[INFO] #RESTRICTED truth types = {n_true}")
    assert n_true == 216

    true_vals = np.array(
        [np.array(ranks_to_vals(r, cfg.M, cfg.T), float) for r in true_ranks],
        dtype=float
    )

    # ---------------------------------------------
    # 2) TypeCx（厳密順序 720）
    # ---------------------------------------------
    cx_ranks = generate_strict_orders_6()
    n_cx = len(cx_ranks)

    print(f"[INFO] #TypeCx ranks = {n_cx}")
    assert n_cx == 720

    cx_vals = np.array(
        [np.array(ranks_to_vals(r, cfg.M, cfg.T), float) for r in cx_ranks],
        dtype=float
    )

    # ---------------------------------------------
    # 3) truth–truth の期待配分（216^2）
    # ---------------------------------------------
    EA_truth = np.zeros((n_true, n_true, cfg.A, cfg.M, cfg.T), dtype=float)

    for i in tqdm(range(n_true), desc="Precompute E[alloc] truth-truth (216^2)"):
        for j in range(n_true):
            rv = np.zeros((cfg.A, cfg.M, cfg.T), dtype=float)
            rv[0] = true_vals[i]
            rv[1] = true_vals[j]
            EA_truth[i, j] = expected_alloc_given_reports(cfg, rv)

    honest_U_A = np.zeros((n_true, n_true), dtype=float)
    for i in range(n_true):
        for j in range(n_true):
            honest_U_A[i, j] = utility(EA_truth[i, j, 0], true_vals[i])

    # ---------------------------------------------
    # 4) Cx–truth の期待配分（A が嘘）
    # ---------------------------------------------
    # EA_CxA[k, j, m, t] = E[alloc(A,m,t) | reportA=Cx(k), reportB=true(j)]
    EA_CxA = np.zeros((n_cx, n_true, cfg.M, cfg.T), dtype=float)

    for k in tqdm(range(n_cx), desc="Precompute E[alloc] (A: Cx, B: truth) 720x216"):
        for j in range(n_true):
            rv = np.zeros((cfg.A, cfg.M, cfg.T), dtype=float)
            rv[0] = cx_vals[k]
            rv[1] = true_vals[j]
            EA = expected_alloc_given_reports(cfg, rv)
            EA_CxA[k, j] = EA[0]

    # ---------------------------------------------
    # 5) best response（A 側のみ）
    # ---------------------------------------------
    bestCx_U_A = np.zeros((n_true, n_true), dtype=float)
    bestCx_idx_A = np.zeros((n_true, n_true), dtype=int)

    for i in tqdm(range(n_true), desc="Best response TypeCx (A only)"):
        tvA_flat = true_vals[i].reshape(-1)
        for j in range(n_true):
            mat = EA_CxA[:, j].reshape(n_cx, -1)  # (720,6)
            utils = mat @ tvA_flat
            k_star = int(np.argmax(utils))
            bestCx_U_A[i, j] = float(utils[k_star])
            bestCx_idx_A[i, j] = k_star

    # ---------------------------------------------
    # 6) 出力
    # ---------------------------------------------
    rows = []
    for i in range(n_true):
        for j in range(n_true):
            honestA = honest_U_A[i, j]
            bestA = bestCx_U_A[i, j]
            rows.append({
                "true_A": i,
                "true_B": j,
                "honest_A": honestA,
                "bestU_A_typeCx": bestA,
                "best_reportA_typeCx": int(bestCx_idx_A[i, j]),
                "gainA_typeCx": bestA - honestA,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ic_summary_typeCx.csv"), index=False)

    viol = df[df["gainA_typeCx"] > 1e-9]
    viol.to_csv(os.path.join(out_dir, "ic_violations_typeCx.csv"), index=False)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "Config": asdict(cfg),
            "ICConfig": asdict(icfg),
            "n_truth": n_true,
            "n_typeCx": n_cx,
            "note": "Unilateral deviation only: A-side TypeCx, B always truthful.",
        }, f, indent=2)

    np.savez_compressed(
        os.path.join(out_dir, "tables.npz"),
        honest_U_A=honest_U_A,
        bestCx_U_A=bestCx_U_A,
        bestCx_idx_A=bestCx_idx_A,
    )

    print(f"[DONE] results saved to {out_dir}")
