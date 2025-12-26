from __future__ import annotations

"""
実験A（Type A / Type B）

- 真の選好（RESTRICTED = 216）を全列挙
- Phase A（RSD + window + buffer + 量モデル）を用いる
- allocation は report により決定
- utility は true type により評価
- outside は「割当が起きなければ 0」
  （outside スロットに割当が起きた場合のみ -1 が効く）

出力：
- 正直申告時の期待効用
- Type A / Type B 嘘を許したときの
  ・最大期待効用
  ・その効用を達成する report index
  ・gain（正直との差）
"""

import os
import json
import numpy as np
import pandas as pd
from dataclasses import asdict
from tqdm import tqdm

from src.config.base_config import Config, ICConfig
from src.core.generator import (
    generate_consistent_with_truncation,
    generate_typeA_tie_break_lies,
    ranks_to_vals,
)
from src.core.initial_rsd import run_phaseA_given_orders
from src.utils.results import make_timestamped_dir


# ------------------------------------------------------------
# 内部関数
# ------------------------------------------------------------

def expected_alloc_given_reports(cfg: Config, report_vals_pair: np.ndarray) -> np.ndarray:
    """
    report_vals_pair (A,M,T) が固定されたときの
    Phase A の期待配分 E[alloc] を order_seq 全列挙で計算。

    return: np.ndarray (A,M,T), float
    """
    assert cfg.A == 2
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
    utility = sum_t,m E[alloc] * true_vals
    （未割当は 0、outside スロットは true_vals = -1）
    """
    return float(np.sum(expected_alloc_mt * true_vals_mt))


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------

def run():
    cfg = Config()
    icfg = ICConfig()

    assert cfg.A == 2 and cfg.M == 2 and cfg.T == 3
    assert cfg.STRATEGY_DOMAIN == "RESTRICTED"

    out_dir = make_timestamped_dir(icfg.OUTPUT_BASE_DIR, prefix="expA_AB_with_best_report")
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. 真の選好（RESTRICTED = 216）
    # --------------------------------------------------------
    restricted = generate_consistent_with_truncation(cfg.M, cfg.T)
    ranks_list = [d["ranks"] for d in restricted]
    n = len(ranks_list)
    print(f"[INFO] #RESTRICTED types = {n}")
    assert n == 216

    type_vals = np.array(
        [np.array(ranks_to_vals(r, cfg.M, cfg.T), float) for r in ranks_list]
    )  # (n,M,T)

    idx_map = {r: k for k, r in enumerate(ranks_list)}

    # --------------------------------------------------------
    # 2. (reportA, reportB) ごとの期待配分を全キャッシュ
    # --------------------------------------------------------
    EA = np.zeros((n, n, cfg.A, cfg.M, cfg.T), dtype=float)

    for rA in tqdm(range(n), desc="Precompute E[alloc] for report pairs"):
        for rB in range(n):
            rv = np.zeros((cfg.A, cfg.M, cfg.T), dtype=float)
            rv[0] = type_vals[rA]
            rv[1] = type_vals[rB]
            EA[rA, rB] = expected_alloc_given_reports(cfg, rv)

    EA_A = EA[:, :, 0]  # (n,n,M,T)
    EA_B = EA[:, :, 1]

    # --------------------------------------------------------
    # 3. 正直効用
    # --------------------------------------------------------
    honest_U_A = np.zeros((n, n))
    honest_U_B = np.zeros((n, n))

    for i in range(n):
        tvA = type_vals[i]
        for j in range(n):
            tvB = type_vals[j]
            honest_U_A[i, j] = utility(EA_A[i, j], tvA)
            honest_U_B[i, j] = utility(EA_B[i, j], tvB)

    # --------------------------------------------------------
    # 4. Type B（別の一貫した選好を丸ごと報告）
    # --------------------------------------------------------
    bestB_U_A = np.zeros((n, n))
    bestB_U_B = np.zeros((n, n))
    bestB_idx_A = np.zeros((n, n), dtype=int)
    bestB_idx_B = np.zeros((n, n), dtype=int)

    # A側
    for i in tqdm(range(n), desc="Best response TypeB (A)"):
        tvA_flat = type_vals[i].reshape(-1)
        for j in range(n):
            mat = EA_A[:, j].reshape(n, -1)
            utils = mat @ tvA_flat
            k_star = int(np.argmax(utils))
            bestB_U_A[i, j] = float(utils[k_star])
            bestB_idx_A[i, j] = k_star

    # B側
    for j in tqdm(range(n), desc="Best response TypeB (B)"):
        tvB_flat = type_vals[j].reshape(-1)
        for i in range(n):
            mat = EA_B[i, :].reshape(n, -1)
            utils = mat @ tvB_flat
            k_star = int(np.argmax(utils))
            bestB_U_B[i, j] = float(utils[k_star])
            bestB_idx_B[i, j] = k_star

    # --------------------------------------------------------
    # 5. Type A（無差別部分の tie-breaking）
    # --------------------------------------------------------
    bestA_U_A = np.zeros((n, n))
    bestA_U_B = np.zeros((n, n))
    bestA_idx_A = np.zeros((n, n), dtype=int)
    bestA_idx_B = np.zeros((n, n), dtype=int)

    # A側
    for i in tqdm(range(n), desc="Best response TypeA (A)"):
        tvA = type_vals[i]
        cand_ranks = generate_typeA_tie_break_lies(ranks_list[i], cfg.M, cfg.T)
        cand_idx = [idx_map[r] for r in cand_ranks if r in idx_map]
        if not cand_idx:
            cand_idx = [i]

        for j in range(n):
            best_val = honest_U_A[i, j]
            best_k = i
            for k in cand_idx:
                u = utility(EA_A[k, j], tvA)
                if u > best_val:
                    best_val = u
                    best_k = k
            bestA_U_A[i, j] = best_val
            bestA_idx_A[i, j] = best_k

    # B側
    for j in tqdm(range(n), desc="Best response TypeA (B)"):
        tvB = type_vals[j]
        cand_ranks = generate_typeA_tie_break_lies(ranks_list[j], cfg.M, cfg.T)
        cand_idx = [idx_map[r] for r in cand_ranks if r in idx_map]
        if not cand_idx:
            cand_idx = [j]

        for i in range(n):
            best_val = honest_U_B[i, j]
            best_k = j
            for k in cand_idx:
                u = utility(EA_B[i, k], tvB)
                if u > best_val:
                    best_val = u
                    best_k = k
            bestA_U_B[i, j] = best_val
            bestA_idx_B[i, j] = best_k

    # --------------------------------------------------------
    # 6. CSV 出力
    # --------------------------------------------------------
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({
                "true_A": i,
                "true_B": j,

                "honest_A": honest_U_A[i, j],
                "honest_B": honest_U_B[i, j],

                # --- Type A ---
                "bestU_A_typeA": bestA_U_A[i, j],
                "best_reportA_typeA": bestA_idx_A[i, j],
                "gainA_typeA": bestA_U_A[i, j] - honest_U_A[i, j],

                "bestU_B_typeA": bestA_U_B[i, j],
                "best_reportB_typeA": bestA_idx_B[i, j],
                "gainB_typeA": bestA_U_B[i, j] - honest_U_B[i, j],

                # --- Type B ---
                "bestU_A_typeB": bestB_U_A[i, j],
                "best_reportA_typeB": bestB_idx_A[i, j],
                "gainA_typeB": bestB_U_A[i, j] - honest_U_A[i, j],

                "bestU_B_typeB": bestB_U_B[i, j],
                "best_reportB_typeB": bestB_idx_B[i, j],
                "gainB_typeB": bestB_U_B[i, j] - honest_U_B[i, j],
            })



    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ic_summary.csv"), index=False)

    # --------------------------------------------------------
    # 7. IC violation の抽出
    # --------------------------------------------------------
    viol = df[
        (df["gainA_typeA"] > 1e-9) |
        (df["gainA_typeB"] > 1e-9) |
        (df["gainB_typeA"] > 1e-9) |
        (df["gainB_typeB"] > 1e-9)
    ]

    viol.to_csv(
        os.path.join(out_dir, "ic_violations.csv"),
        index=False
    )


    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "Config": asdict(cfg),
            "ICConfig": asdict(icfg),
            "n_types": n,
            "lie_types": ["TypeA", "TypeB"],
        }, f, indent=2)

    print(f"[DONE] results saved to {out_dir}")
