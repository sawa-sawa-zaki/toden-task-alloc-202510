#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IC（誘因）検証：2マシン×3時間。
- RSD は t=0,1 のみを対象（t=2 は初期配分なし）
- TTC 調整で t=2 に流入し得る
- 真の選好（truth）：時間単調＋マシン間一貫、順位同型は重複排除、-1（アウトサイド）を別扱い
- 申告（report）：一貫性不要（時間逆転もマシン間逆転も可）、-1を許容
- ショックは t=0,1 のみで発生、t=2 は 0 固定
"""

from __future__ import annotations

# --- プロジェクトパス追加 ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import csv
import argparse
from itertools import product
import numpy as np
from types import SimpleNamespace

from tqdm import tqdm  # 進捗バー（pip install tqdm）

from src.allocsim.market import Market
from src.allocsim.process import run_rsd_only, run_ttc_only
from src.allocsim.utils import seed_all

# ========== 固定パラメータ ==========
A, M, T = 2, 2, 3
UNITS_DEMAND_EACH = 1
MACHINE_CAPACITY = [1, 1]   # per machine per time
FORECAST_BY_T    = [1, 1, 1]  # t=2 にもキャパは用意（初期RSDでは使わない）
OUTSIDE = -1

# 真の選好・申告の値レベル
LEVELS_TRUE   = [-1, 1, 10, 100, 1000]
LEVELS_REPORT = [-1, 1, 10, 100, 1000]


OUTSIDE = -1  # 既にこの定義があるなら重複しないように

def _sign(x: float) -> int:
    if x > 0: return 1
    if x < 0: return -1
    return 0

def _time_ok(e, l):
    # 早い→遅い（t0>=t1>=t2）を許し、-1 を含むときの特別扱いを維持
    if e == OUTSIDE and l != OUTSIDE: return False
    if e != OUTSIDE and l == OUTSIDE: return True
    if e == OUTSIDE and l == OUTSIDE: return True
    return e >= l

def _signature_2x3(v: np.ndarray) -> tuple:
    """
    v: (M=2,T=3)。順位だけを符号化して重複を潰す。
    -1 は -1 に固定。それ以外の正値は降順にランク付けして {3,2,1,0} に写像。
    戻り: (A@t0, A@t1, A@t2, B@t0, B@t1, B@t2)
    """
    assert v.shape == (2,3)
    slots = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    vals = [float(v[m,t]) for (m,t) in slots]
    outs = [x == OUTSIDE for x in vals]
    pos = sorted({x for x in vals if x != OUTSIDE}, reverse=True)
    rank_map = {}
    for idx, x in enumerate(pos):
        # 高い値ほど大きいランク（3→2→1→0）
        rank_map[x] = max(0, 3 - idx)
    sig = []
    for x, is_out in zip(vals, outs):
        sig.append(-1 if is_out else rank_map[x])
    return tuple(sig)


# ====== 真の選好（時間単調＋マシン間一貫、順位同型は重複排除） ======
def _sign(x: float) -> int:
    if x > 0: return 1
    if x < 0: return -1
    return 0

def _time_ok(e, l):
    # early=-1 かつ late> -1 は逆転→不可、late=-1 はOK
    if e == OUTSIDE and l != OUTSIDE: return False
    if e != OUTSIDE and l == OUTSIDE: return True
    if e == OUTSIDE and l == OUTSIDE: return True
    return e >= l

def _signature_2x3(v: np.ndarray) -> tuple:
    """
    v: (M=2,T=3)。-1は最弱。正値は相対順位（高=3, 次=2, 次=1, 低=0）で符号化。
    戻り値は (A@t0, A@t1, A@t2, B@t0, B@t1, B@t2)
    """
    assert v.shape == (2,3)
    slots = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    vals = [float(v[m,t]) for (m,t) in slots]
    outs = [x == OUTSIDE for x in vals]
    pos = sorted({x for x in vals if x != OUTSIDE}, reverse=True)
    rank_map = {}
    for idx, x in enumerate(pos):
        rank_map[x] = max(0, 3 - idx)  # 3,2,1,0 の4段階
    sig = []
    for x, is_out in zip(vals, outs):
        sig.append(-1 if is_out else rank_map[x])
    return tuple(sig)

def generate_truth_space_2x3():
    L = LEVELS_TRUE
    # 各マシンで (t0,t1,t2) は time monotone（-1の例外規則込み）
    triples = []
    for e0 in L:
        for e1 in L:
            if not _time_ok(e0, e1): continue
            for e2 in L:
                if not _time_ok(e1, e2): continue
                # ここまでで e0>=e1>=e2（-1の扱い含む）
                triples.append((e0,e1,e2))

    seen = set()
    uniq = []
    for (a0,a1,a2), (b0,b1,b2) in product(triples, triples):
        v = np.array([[a0,a1,a2],[b0,b1,b2]], dtype=float)
        # マシン間一貫：A-B の符号が t 全体で反転しない（0は許容）
        d = [v[0,t]-v[1,t] for t in range(3)]
        s = [_sign(x) for x in d]
        # 1 と -1 が混在したら不許可（0はOK）
        if 1 in s and -1 in s:
            continue
        # 順位同型の重複排除
        sig = _signature_2x3(v)
        if sig in seen: continue
        seen.add(sig)
        # 代表値（rank3→100, rank2→10, rank1→1, rank0→1でも良いが4段にする）
        # ここでは rank3=100, rank2=10, rank1=1, rank0=1 として単純化
        rep = []
        for r in sig:
            if r == -1: rep.append(OUTSIDE)
            elif r == 3: rep.append(100.0)
            elif r == 2: rep.append(10.0)
            else:        rep.append(1.0)  # r==1 or 0 は 1 に潰す（順位だけ重要）
        rep_mat = np.array([[rep[0],rep[1],rep[2]],[rep[3],rep[4],rep[5]]], dtype=float)
        uniq.append(rep_mat)
    return uniq


def enumerate_reports_2x3(consistency: str = "both", allow_minus_one: bool = True):
    """
    2台×3時間の報告候補を列挙。
    - consistency:
        "none"     : 無制約（時間逆転・マシン逆転OK）※超重いので注意
        "time"     : 時間単調のみ（各マシンで v[:,t0] >= v[:,t1] >= v[:,t2]）
        "machine"  : マシン間の一貫性のみ（A-B 差の符号が時刻間で反転しない）
        "both"     : 時間単調 ＋ マシン間一貫（推奨）
    - allow_minus_one: -1 を報告に許すか
    戻り値: list[np.ndarray shape (2,3)]
    """
    # 値レベル（順位しか意味がないので 100/10/1 に圧縮。-1 は別扱い）
    POS = [100.0, 10.0, 1.0]
    L = ([OUTSIDE] if allow_minus_one else []) + POS

    def time_monotone_triples(levels):
        triples = []
        for e0 in levels:
            for e1 in levels:
                if not _time_ok(e0, e1): continue
                for e2 in levels:
                    if not _time_ok(e1, e2): continue
                    triples.append((e0,e1,e2))
        return triples

    def all_triples(levels):
        triples = []
        for e0 in levels:
            for e1 in levels:
                for e2 in levels:
                    triples.append((e0,e1,e2))
        return triples

    # 時間単調の適用
    if consistency in ("time", "both"):
        triplesA = time_monotone_triples(L)
        triplesB = time_monotone_triples(L)
    else:
        triplesA = all_triples(L)
        triplesB = all_triples(L)

    seen = set()
    out = []

    for (a0,a1,a2) in triplesA:
        for (b0,b1,b2) in triplesB:
            v = np.array([[a0,a1,a2],[b0,b1,b2]], dtype=float)

            # マシン間一貫（必要なら）
            if consistency in ("machine", "both"):
                d = [v[0,t]-v[1,t] for t in range(3)]
                s = [_sign(x) for x in d]
                if 1 in s and -1 in s:
                    # A-B の優劣が時刻間で反転するのは禁止（0は許容）
                    continue

            # 順位同型の重複を除去（同じ相対順序は1通りに潰す）
            sig = _signature_2x3(v)
            if sig in seen:
                continue
            seen.add(sig)

            # 代表値へ正規化（-1 は -1、それ以外は rank→{100,10,1} に写像）
            rep = []
            for r in sig:
                if r == -1: rep.append(OUTSIDE)
                elif r == 3: rep.append(100.0)
                elif r == 2: rep.append(10.0)
                else:        rep.append(1.0)  # r==1 or 0 をまとめる
            rep_mat = np.array([[rep[0],rep[1],rep[2]],[rep[3],rep[4],rep[5]]], dtype=float)
            out.append(rep_mat)

    return out


# ====== メカニズムの1回実行：RSD(t=0,1) → TTC（t=2 も可） ======
def run_once(values_report_all, buffer_by_t, shocks_vec, rng_seed=0):
    """
    values_report_all: (A,M,T) 申告行列（-1可）
    buffer_by_t: 長さT（例 [0,0,0] or [1,1,0]）
    shocks_vec: 長さT（ここでは t=0,1 のみにランダム、t=2 は 0固定）
    """
    market = Market(
        num_agents=A, num_machines=M, time_horizon=T,
        units_demand=UNITS_DEMAND_EACH,
        supply_forecast_by_t=list(FORECAST_BY_T),
        buffer_by_t=list(buffer_by_t),
        machine_capacity=list(MACHINE_CAPACITY),
        supply_shock_by_t=[{"t": t, "delta": int(shocks_vec[t])} for t in range(T)],
    )
    rng = np.random.default_rng(rng_seed)

    # 1) RSD は t=0,1 のみで実行（t=2 は強制的に -1 扱いになるようマスク）
    rsd_res = run_rsd_only(values_report_all, market, rng, active_times=[0,1])
    alloc_after_rsd = rsd_res["allocation"]

    # 2) TTC 調整で t=2 にも移り得る（values は素の申告を使う）
    ttc_res = run_ttc_only(values_report_all, market, rng, init_alloc=alloc_after_rsd)
    alloc_final = ttc_res["allocation"]
    return alloc_final, {"rsd_logs": rsd_res["logs"], "ttc_logs": ttc_res["logs"]}


# ====== 効用（真の選好で評価） ======
def util_true(true_values_all, alloc, a):
    return float(np.sum(alloc[:, a, :].T * true_values_all[a]))


# ====== ショックのサンプリング（t=0,1 のみランダム, t=2=0） ======
def sample_shocks_2x3(mode: str, trials: int, seed: int):
    rng = np.random.default_rng(seed)
    if mode == "none":
        return np.zeros((trials, T), dtype=int)
    vals = np.array([-1, 0, 1], dtype=int)
    probs = np.array([0.25, 0.5, 0.25], dtype=float) ### 仮に変更中！！
    idx01 = rng.choice(len(vals), size=(trials, 2), p=probs)
    s = vals[idx01]
    out = np.zeros((trials, 3), dtype=int)
    out[:, 0] = s[:, 0]
    out[:, 1] = s[:, 1]
    out[:, 2] = 0  # t=2 はショックなし
    return out


# ====== 期待効用（RSD→TTC を毎回回し、真の選好で評価） ======
def expected_utility(true_values_all, report_values_all, me_idx, buffer, shocks_mode, trials, seed):
    shocks_mat = sample_shocks_2x3(shocks_mode, trials, seed)
    utils = []
    for s in shocks_mat:
        alloc, _ = run_once(report_values_all, buffer_by_t=[buffer]*T, shocks_vec=s, rng_seed=seed)
        utils.append(util_true(true_values_all, alloc, me_idx))
    return float(np.mean(utils))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="results/ic_2x3_rsd12_ttc.csv")
    ap.add_argument("--buffer", type=int, choices=[0,1], default=0)
    ap.add_argument("--shocks", type=str, choices=["none","dist"], default="dist")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--limit_truth", type=int, default=None, help="真実プロファイル数の上限")
    ap.add_argument("--limit_reports", type=int, default=None, help="報告候補数の上限（計算重いとき用）")
    ap.add_argument(
        "--report_consistency",
        choices=["none", "time", "machine", "both"],  # none=無制約 / time=時間単調のみ / machine=マシン間一貫のみ / both=両方
        default="both",
        help="提出メッセージ（嘘）に課す一貫性制約。計算を軽くしたいなら both を推奨。"
    )
    ap.add_argument(
        "--allow_minus_one",
        type=int,
        choices=[0,1],
        default=1,
        help="-1（アウトサイド）を報告で許容するか（1=許可, 0=禁止）。0にすると候補がさらに減る。"
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seed_all(args.seed)

    # 真の選好（時間単調＋マシン間一貫、順位同型は重複排除）
    truth_space = generate_truth_space_2x3()
    truth_profiles = list(product(range(len(truth_space)), repeat=2))
    if args.limit_truth is not None:
        truth_profiles = truth_profiles[:args.limit_truth]

    # 申告（制約あり/なしを切替）
    report_space = enumerate_reports_2x3(
        consistency=args.report_consistency,
        allow_minus_one=bool(args.allow_minus_one)
    )
    if args.limit_reports is not None:
        report_space = report_space[:args.limit_reports]

    print(f"[info] truth_space={len(truth_space)}, report_space={len(report_space)}, profiles={len(truth_profiles)}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "profile_id",
                "i_true_idx","j_true_idx",
                "i_truth_is_BR","i_gain_exp","i_BR_idx",
                "j_truth_is_BR","j_gain_exp","j_BR_idx",
                "buffer","shocks","trials","seed"
            ]
        )
        writer.writeheader()

        for pid, (i_idx, j_idx) in tqdm(enumerate(truth_profiles), total=len(truth_profiles), desc="Profiles"):
            Vi_true = truth_space[i_idx]; Vj_true = truth_space[j_idx]
            true_vals = np.stack([Vi_true, Vj_true], axis=0)

            # baseline（両者 truthful）
            u_i_truth = expected_utility(true_vals, true_vals, me_idx=0,
                                         buffer=args.buffer, shocks_mode=args.shocks,
                                         trials=args.trials, seed=args.seed)
            u_j_truth = expected_utility(true_vals, true_vals, me_idx=1,
                                         buffer=args.buffer, shocks_mode=args.shocks,
                                         trials=args.trials, seed=args.seed)

            # i のBR（jはtruth固定）— 申告は無制約に総当り
            i_best_u = u_i_truth; i_BR_idx = -1
            for ridx, Ri in enumerate(report_space):
                rep_vals = np.stack([Ri, Vj_true], axis=0)
                u = expected_utility(true_vals, rep_vals, me_idx=0,
                                     buffer=args.buffer, shocks_mode=args.shocks,
                                     trials=args.trials, seed=args.seed)
                if u > i_best_u + 1e-12:
                    i_best_u = u; i_BR_idx = ridx
            i_gain = i_best_u - u_i_truth
            i_truth_is_BR = int(i_BR_idx == -1)

            # j のBR（iはtruth固定）
            j_best_u = u_j_truth; j_BR_idx = -1
            for ridx, Rj in enumerate(report_space):
                rep_vals = np.stack([Vi_true, Rj], axis=0)
                u = expected_utility(true_vals, rep_vals, me_idx=1,
                                     buffer=args.buffer, shocks_mode=args.shocks,
                                     trials=args.trials, seed=args.seed)
                if u > j_best_u + 1e-12:
                    j_best_u = u; j_BR_idx = ridx
            j_gain = j_best_u - u_j_truth
            j_truth_is_BR = int(j_BR_idx == -1)

            writer.writerow({
                "profile_id": pid,
                "i_true_idx": i_idx, "j_true_idx": j_idx,
                "i_truth_is_BR": i_truth_is_BR, "i_gain_exp": f"{i_gain:.6f}", "i_BR_idx": i_BR_idx,
                "j_truth_is_BR": j_truth_is_BR, "j_gain_exp": f"{j_gain:.6f}", "j_BR_idx": j_BR_idx,
                "buffer": args.buffer, "shocks": args.shocks, "trials": args.trials, "seed": args.seed,
            })

    print(f"[done] wrote: {args.out}")


if __name__ == "__main__":
    main()
