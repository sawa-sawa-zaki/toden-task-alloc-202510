#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IC（誘因）検証：相手が正直申告のとき、自分のベストレスポンス（BR）を
“制約付きメッセージ空間”から探索。配分は申告で決まるが、効用評価は
常に“真の選好（truth）”で行う。

本版のポイント：
- 需要=1, A=2, M=2, T=2
- 真の選好（truth）値レベルは { -1, 1, 10, 100, 1000 }。
  * 真の選好にも一貫性を課す（時間単調 + マシン間の優劣一貫）。
  * “順位（相対順序）だけ”を用いて重複を排除（例：[[100,100],[10,10]] と [[100,100],[1,1]] は同一）。
  * -1 はアウトサイド（割当禁止）として別扱い（どの正の値よりも劣る）。
- 申告メッセージ（report）も時間単調 + マシン間一貫（-1 を既定で許可）。
- 予測供給= [1,1]、マシン容量= [1,1]、ショックは none または dist（{-1,0,1}に確率{1/4,1/2,1/4}）。
- バッファは 0/1。
- 出力CSV：各真実プロファイルごとに、truth がBRか（誘因の有無）、期待利得、BRインデックス等を記録。

実行例：
  # ベンチマーク：変動なし（RSDのみ）
  python scripts/check_incentives_best_response.py --seed 42 --buffer 0 --shocks none \
      --out results/ic_benchmark_no_shock.csv

  # 変動あり・バッファ0
  python scripts/check_incentives_best_response.py --seed 42 --buffer 0 --shocks dist \
      --trials 2000 --out results/ic_shock_buf0.csv

  # 変動あり・バッファ1
  python scripts/check_incentives_best_response.py --seed 42 --buffer 1 --shocks dist \
      --trials 2000 --out results/ic_shock_buf1.csv
"""
from __future__ import annotations

# --- プロジェクトルートをパスに追加（src/allocsim を import 可能にする） ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------------------------------

import os
import csv
import argparse
from itertools import product
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm 

from src.allocsim.market import Market
from src.allocsim.process import run_pipeline_for_one_seed
from src.allocsim.utils import seed_all

# ===== 固定条件（検証1） =====
A, M, T = 2, 2, 2
UNITS_DEMAND_EACH = 1
MACHINE_CAPACITY = [1, 1]   # per machine per time
FORECAST_BY_T    = [1, 1]   # per time

# 値レベル
LEVELS_TRUE   = [-1, 1, 10, 100, 1000]  # 真の選好（-1=アウトサイド）
LEVELS_REPORT = [-1, 1, 10, 100, 1000]  # 申告（-1=アウトサイド）

OUTSIDE = -1  # アウトサイド（割当禁止）


# ================= 真の選好：重複排除付きの一貫性あり生成 =================

def _sign_for_machine_consistency(x: float) -> int:
    """
    マシン間一貫性の符号判定用。OUTSIDE=-1 はどの正値よりも劣る。
    返り値: 1(正), -1(負), 0(同値)
    """
    if x > 0: return 1
    if x < 0: return -1
    return 0


def _order_signature_2x2(v: np.ndarray) -> tuple:
    """
    v: (M=2, T=2)。-1 は OUTSIDE。
    数値の“相対順序”と“同値”だけを signature にエンコードする。
    - OUTSIDE は -1 のまま（最弱クラス）
    - 残りの正値は行列内の相対順位で符号化（高=2, 次=1, 低=0；同値は同ランク）
    戻り値はスロット順 (A@early, A@late, B@early, B@late) の整数タプル。
    """
    assert v.shape == (2, 2)
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    vals = [float(v[m, t]) for (m, t) in slots]

    outside_mask = [x == OUTSIDE for x in vals]
    pos_vals_sorted_desc = sorted({x for x in vals if x != OUTSIDE}, reverse=True)

    rank_map = {}
    for idx, x in enumerate(pos_vals_sorted_desc):
        # 最大3ランク（2,1,0）で十分（ユニーク値が3超でも順に割り当て）
        rank = max(0, 2 - idx)
        rank_map[x] = rank

    sig = []
    for x, is_out in zip(vals, outside_mask):
        if is_out:
            sig.append(-1)
        else:
            sig.append(rank_map[x])
    return tuple(sig)


def generate_truth_space():
    """
    真の選好（需要=1）を LEVELS_TRUE から生成。
    一貫性（時間単調 + マシン間の優劣一貫）を満たすものに限定し、
    さらに“順位だけ”同じケースは signature で重複排除。
    代表値として {rank2=100, rank1=10, rank0=1, OUTSIDE=-1} に正規化して返す。
    """
    L = LEVELS_TRUE

    def time_ok(e, l):
        # 早い=-1 で遅いが有効値は逆転 → 不許可 / 遅い=-1 はOK / 共に-1もOK
        if e == OUTSIDE and l != OUTSIDE: return False
        if e != OUTSIDE and l == OUTSIDE: return True
        if e == OUTSIDE and l == OUTSIDE: return True
        return e >= l

    pairs = [(e, l) for e in L for l in L if time_ok(e, l)]

    seen = set()
    uniq = []

    for (a_e, a_l), (b_e, b_l) in product(pairs, pairs):
        v = np.array([[a_e, a_l], [b_e, b_l]], dtype=float)

        # マシン間一貫性：A vs B の優劣符号が全時刻で一致（0は許容）
        d0 = v[0, 0] - v[1, 0]   # t=early
        d1 = v[0, 1] - v[1, 1]   # t=late
        s0, s1 = _sign_for_machine_consistency(d0), _sign_for_machine_consistency(d1)
        if s0 * s1 == -1:
            continue

        sig = _order_signature_2x2(v)
        if sig in seen:
            continue
        seen.add(sig)

        # 代表値に正規化：-1 は OUTSIDE、それ以外は {2→100, 1→10, 0→1}
        rep = []
        for r in sig:
            if r == -1: rep.append(OUTSIDE)
            elif r == 2: rep.append(100.0)
            elif r == 1: rep.append(10.0)
            else:        rep.append(1.0)
        rep_mat = np.array([[rep[0], rep[1]], [rep[2], rep[3]]], dtype=float)
        uniq.append(rep_mat)

    return uniq
# ==========================================================================


# ================= 申告メッセージ空間（時間単調＋マシン間一貫） ==============
def enumerate_reports(consistency: str = "both_time_and_machine", allow_minus_one: bool = True):
    levels = LEVELS_REPORT if allow_minus_one else [x for x in LEVELS_REPORT if x != OUTSIDE]

    def time_ok(e, l):
        if allow_minus_one:
            if e == OUTSIDE and l != OUTSIDE: return False
            if e != OUTSIDE and l == OUTSIDE: return True
            if e == OUTSIDE and l == OUTSIDE: return True
            return e >= l
        else:
            return e >= l

    pairs = [(e, l) for e in levels for l in levels if time_ok(e, l)]

    cand = []
    for (a_e, a_l), (b_e, b_l) in product(pairs, pairs):
        v = np.array([[a_e, a_l], [b_e, b_l]], dtype=float)
        if consistency == "both_time_and_machine":
            def sign(x):
                if x > 0: return 1
                if x < 0: return -1
                return 0
            d0 = v[0, 0] - v[1, 0]
            d1 = v[0, 1] - v[1, 1]
            if sign(d0) * sign(d1) == -1:
                continue
        cand.append(v)
    return cand
# ==========================================================================


# ================= メカニズム実行・効用評価（真の選好で評価） ================
def run_once(report_values_all, buffer_by_t, shocks):
    """
    report_values_all: (A,M,T) 申告値（-1 あり）
    buffer_by_t: 長さT（例: [0,0] または [1,1]）
    shocks: 長さT（例: [-1,0,1]）
    """
    market = Market(
        num_agents=A, num_machines=M, time_horizon=T,
        units_demand=UNITS_DEMAND_EACH,
        supply_forecast_by_t=list(FORECAST_BY_T),
        buffer_by_t=list(buffer_by_t),
        machine_capacity=list(MACHINE_CAPACITY),
        supply_shock_by_t=[{"t": t, "delta": int(shocks[t])} for t in range(T)],
    )
    cfg = SimpleNamespace(
        pipeline=["rsd_initial_global", "ttc_global"],
        market=SimpleNamespace(units_demand=UNITS_DEMAND_EACH),
    )
    rng = np.random.default_rng(0)  # RSD順序乱択（必要なら引数化）
    res = run_pipeline_for_one_seed(cfg, report_values_all.astype(float), market, rng)
    return res["allocation"]  # (T, A, M)


def util_true(true_values_all, alloc, a):
    """常に“真の選好”で評価。"""
    return float(np.sum(alloc[:, a, :].T * true_values_all[a]))


def sample_shocks(mode: str, trials: int, seed: int):
    """ mode='none' で 0 固定、'dist' で {-1,0,1} に確率 {1/4,1/2,1/4} """
    rng = np.random.default_rng(seed)
    if mode == "none":
        return np.zeros((trials, T), dtype=int)
    vals = np.array([-1, 0, 1], dtype=int)
    probs = np.array([0.25, 0.5, 0.25], dtype=float)
    idx = rng.choice(len(vals), size=(trials, T), p=probs)
    return vals[idx]


def expected_utility(true_values_all, report_values_all, me_idx, buffer, shocks_mode, trials, seed):
    """配分は report、評価は true で Monte Carlo 平均。"""
    shocks_mat = sample_shocks(shocks_mode, trials, seed)
    utils = []
    for s in shocks_mat:
        alloc = run_once(report_values_all, buffer_by_t=[buffer]*T, shocks=s)
        utils.append(util_true(true_values_all, alloc, me_idx))
    return float(np.mean(utils))
# ==========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="results/ic_result.csv")
    ap.add_argument("--buffer", type=int, choices=[0, 1], default=0)
    ap.add_argument("--shocks", type=str, choices=["none", "dist"], default="none")
    ap.add_argument("--trials", type=int, default=2000, help="shocks=dist のときのMC試行数")
    ap.add_argument("--report_consistency", type=str,
                    choices=["time_monotone_only", "both_time_and_machine"],
                    default="both_time_and_machine")
    ap.add_argument("--allow_minus_one", action="store_true", default=True,
                    help="申告に -1（アウトサイド）を許す（既定: 許可）")
    ap.add_argument("--limit", type=int, default=None, help="真実プロファイル数の上限（デバッグ用）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seed_all(args.seed)

    # 真の選好（重複排除・一貫性あり）
    truth_space = generate_truth_space()
    truth_profiles = list(product(range(len(truth_space)), repeat=2))
    if args.limit is not None:
        truth_profiles = truth_profiles[:args.limit]

    # 申告メッセージ空間（時間単調＋マシン間一貫）
    report_space = enumerate_reports(args.report_consistency, allow_minus_one=args.allow_minus_one)
    report_space = [np.array(v, dtype=float) for v in report_space]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "profile_id",
                "i_true_idx", "j_true_idx",
                "i_truth_is_BR", "i_gain_exp", "i_BR_idx",
                "j_truth_is_BR", "j_gain_exp", "j_BR_idx",
                "buffer", "shocks", "trials", "report_consistency",
                "allow_minus_one", "seed",
            ],
        )
        writer.writeheader()

        for pid, (i_true_idx, j_true_idx) in tqdm(
            enumerate(truth_profiles),
            total=len(truth_profiles),
            desc="Running profiles"
        ):
            Vi_true = truth_space[i_true_idx]
            Vj_true = truth_space[j_true_idx]
            true_vals = np.stack([Vi_true, Vj_true], axis=0)

            # truthful baseline（両者 truth）
            u_i_truth = expected_utility(true_values_all=true_vals,
                                         report_values_all=true_vals,
                                         me_idx=0, buffer=args.buffer,
                                         shocks_mode=args.shocks, trials=args.trials, seed=args.seed)
            u_j_truth = expected_utility(true_values_all=true_vals,
                                         report_values_all=true_vals,
                                         me_idx=1, buffer=args.buffer,
                                         shocks_mode=args.shocks, trials=args.trials, seed=args.seed)

            # i のBR（j は truth 固定）
            i_best_u = u_i_truth
            i_BR_idx = -1  # -1 → truthful がBR
            for ridx, Ri in enumerate(report_space):
                rep_vals = np.stack([Ri, Vj_true], axis=0)
                u = expected_utility(true_values_all=true_vals,
                                     report_values_all=rep_vals,
                                     me_idx=0, buffer=args.buffer,
                                     shocks_mode=args.shocks, trials=args.trials, seed=args.seed)
                if u > i_best_u + 1e-12:
                    i_best_u = u
                    i_BR_idx = ridx
            i_gain = i_best_u - u_i_truth
            i_truth_is_BR = int(i_BR_idx == -1)

            # j のBR（i は truth 固定）
            j_best_u = u_j_truth
            j_BR_idx = -1
            for ridx, Rj in enumerate(report_space):
                rep_vals = np.stack([Vi_true, Rj], axis=0)
                u = expected_utility(true_values_all=true_vals,
                                     report_values_all=rep_vals,
                                     me_idx=1, buffer=args.buffer,
                                     shocks_mode=args.shocks, trials=args.trials, seed=args.seed)
                if u > j_best_u + 1e-12:
                    j_best_u = u
                    j_BR_idx = ridx
            j_gain = j_best_u - u_j_truth
            j_truth_is_BR = int(j_BR_idx == -1)

            writer.writerow({
                "profile_id": pid,
                "i_true_idx": i_true_idx, "j_true_idx": j_true_idx,
                "i_truth_is_BR": i_truth_is_BR, "i_gain_exp": f"{i_gain:.6f}", "i_BR_idx": i_BR_idx,
                "j_truth_is_BR": j_truth_is_BR, "j_gain_exp": f"{j_gain:.6f}", "j_BR_idx": j_BR_idx,
                "buffer": args.buffer, "shocks": args.shocks, "trials": args.trials,
                "report_consistency": args.report_consistency,
                "allow_minus_one": int(args.allow_minus_one),
                "seed": args.seed,
            })

    print(f"[done] wrote: {args.out}")


if __name__ == "__main__":
    main()
