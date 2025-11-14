#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T=2/3 の mode=A（真も嘘も一貫性ドメイン）検証ランチャ。
- 依存：allocsim.online_rsd_ttc.Config / evaluate_ic
- 値生成：allocsim.report_from_values.build_lexicographic_values
- 真の選好の列挙：allocsim.strategy.time_monotone.enumerate_truth_profiles_pair
- 嘘（mode=A）の列挙：allocsim.strategy.time_monotone.enumerate_devs_modeA_for_pair
"""

import os
import sys
import argparse
import json
import numpy as np

# src を import パスに追加
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from allocsim.online_rsd_ttc import Config, evaluate_ic
from allocsim.report_from_values import build_lexicographic_values
from allocsim.strategy.time_monotone import (
    enumerate_truth_profiles_pair,
    enumerate_devs_modeA_for_pair,  # 片側だけが mode-A の全嘘（12通り）を返すユーティリティ
)

def make_unique_outdir(base="results", prefix="global_T_A"):
    import random
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    rnd = random.randint(10000, 99999)
    outdir = os.path.join(base, f"{prefix}_{ts}_{rnd}")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def parse_rank_order(s: str):
    # "(0,1),(1,1),(0,2),(1,2),(0,3),(1,3)" -> [(0,1),...]
    ro = []
    for chunk in s.split(")"):
        chunk = chunk.strip().strip(",")
        if not chunk:
            continue
        tup = chunk.lstrip("(")
        m_str, t1_str = tup.split(",")
        ro.append((int(m_str), int(t1_str)))
    return ro

def parse_ties(s: str):
    # "[(m,t1,g),...]" -> {(m,t1): g}
    if not s or not s.strip():
        return {}
    s = s.strip().lstrip("[").rstrip("]")
    out = {}
    for part in s.split(")"):
        part = part.strip().strip(",")
        if not part:
            continue
        p2 = part.lstrip("(")
        m_str, t1_str, g_str = [x.strip() for x in p2.split(",")]
        out[(int(m_str), int(t1_str))] = int(g_str)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--q", nargs="+", type=int, default=[1, 1])
    ap.add_argument("--barS", nargs="+", type=int, default=[1, 1, 1])
    ap.add_argument("--buffer", nargs="+", type=int, default=[0, 1, 1])
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--use_julia", action="store_true", help="Use Julia backend for per-trial core")

    # 値（効用）生成のランク指定（“時間が早いほど嬉しい”＋A優先のデフォルト）
    ap.add_argument("--rank_order", type=str,
                    default="(0,1),(1,1),(0,2),(1,2),(0,3),(1,3)")
    ap.add_argument("--ties", type=str, default="")

    args = ap.parse_args()
    outdir = args.out_dir or make_unique_outdir(prefix=f"global_T{args.T}_A")
    os.makedirs(outdir, exist_ok=True)

    cfg = Config(
        A=args.A, M=args.M, T=args.T,
        q=args.q, barS=args.barS, buffer=args.buffer,
        window=args.window, shock_p=args.p
    )

    # 値（効用）：レキシコで“時間早いほど嬉しい”
    ro = parse_rank_order(args.rank_order)
    tie_map = parse_ties(args.ties)
    Qmax = max(cfg.q)
    from allocsim.report_from_values import build_lexicographic_values as build_vals
    values = build_vals(
        A=cfg.A, M=cfg.M, T=cfg.T,
        rank_order=ro, Qmax=Qmax,
        ties=tie_map or None,
        agent_bias=[0.0]*cfg.A,
    )

    # 2人の真の選好（Aドメイン）直積：T=3なら 12×12=144
    # enumerate_truth_profiles_pair は
    #   -> (label_pair: {"p1":{machine_order,tau},"p2":{...}}, reports_truth:[report_p1,report_p2])
    truth_pairs = list(enumerate_truth_profiles_pair(cfg.M, cfg.T))

    # devs 列の作成：各 truth に対して「片側だけが mode-A 全嘘（12通り）」を並べる。
    # evaluate_ic は (truth_list, dev_list) を zip で受け取る設計なので、
    # truth/dev を同じ長さのリストに展開して渡す。
    truth_list = []
    dev_list = []
    pair_id = 0
    for label_pair, reports_truth in truth_pairs:
        pair_id += 1
        # 左が嘘
        for label_dev, reports_dev in enumerate_devs_modeA_for_pair(reports_truth, who=0, M=cfg.M, T=cfg.T):
            truth_list.append(reports_truth)
            dev_list.append(reports_dev)
        # 右が嘘
        for label_dev, reports_dev in enumerate_devs_modeA_for_pair(reports_truth, who=1, M=cfg.M, T=cfg.T):
            truth_list.append(reports_truth)
            dev_list.append(reports_dev)

    # ε-IC 評価
    out = evaluate_ic(
        cfg, values, truth_list, dev_list,
        samples=args.samples, seed0=args.seed,
        use_julia=args.use_julia
    )

    # 出力（サマリ＋CSV）
    csv_path = os.path.join(outdir, "ic_global_A.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("pair_id,mean_regret,max_regret\n")
        # 注意：ここでの pair_id は truth_pairs の id とは一致しない（dev を2×12倍しているため）
        # 必要なら後で mapping を拡張してください。
        for k, mean_vec, max_vec in out["results"]:
            f.write(f"{k},{float(np.mean(mean_vec))},{float(np.max(max_vec))}\n")

    print("\n=== Summary ===")
    print(f"Profiles evaluated       : {len(truth_pairs)}")
    print(f"Average regret_mean      : {out['regret_mean_avg']:.6f}")
    print(f"Worst-case regret_max    : {out['regret_max']:.6f}")
    print(f"CSV saved to             : {csv_path}")
    print(f"outdir                   : {outdir}")

if __name__ == "__main__":
    main()
