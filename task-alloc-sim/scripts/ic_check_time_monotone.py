#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
時間単調（早いほど嬉しい）前提での ε-IC を mode=A でチェック。
- 依存：allocsim.online_rsd_ttc.Config / evaluate_ic
- 値生成：allocsim.report_from_values.build_lexicographic_values
- 真の選好：enumerate_truth_profiles_pair（Aドメイン）
- 嘘：enumerate_devs_modeA_for_pair（Aドメイン）
"""

import os
import sys
import argparse
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from allocsim.online_rsd_ttc import Config, evaluate_ic
from allocsim.report_from_values import build_lexicographic_values
from allocsim.strategy.time_monotone import (
    enumerate_truth_profiles_pair,
    enumerate_devs_modeA_for_pair,
)

def parse_rank_order(s: str):
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
    ap.add_argument("--use_julia", action="store_true")

    ap.add_argument("--rank_order", type=str,
                    default="(0,1),(1,1),(0,2),(1,2),(0,3),(1,3)")
    ap.add_argument("--ties", type=str, default="")

    args = ap.parse_args()

    cfg = Config(
        A=args.A, M=args.M, T=args.T,
        q=args.q, barS=args.barS, buffer=args.buffer,
        window=args.window, shock_p=args.p
    )

    ro = parse_rank_order(args.rank_order)
    tie_map = parse_ties(args.ties)
    Qmax = max(cfg.q)
    values = build_lexicographic_values(
        A=cfg.A, M=cfg.M, T=cfg.T,
        rank_order=ro, Qmax=Qmax,
        ties=tie_map or None,
        agent_bias=[0.0]*cfg.A,
    )

    truth_pairs = list(enumerate_truth_profiles_pair(cfg.M, cfg.T))

    truth_list = []
    dev_list = []
    for label_pair, reports_truth in truth_pairs:
        # 左が嘘
        for label_dev, reports_dev in enumerate_devs_modeA_for_pair(reports_truth, who=0, M=cfg.M, T=cfg.T):
            truth_list.append(reports_truth)
            dev_list.append(reports_dev)
        # 右が嘘
        for label_dev, reports_dev in enumerate_devs_modeA_for_pair(reports_truth, who=1, M=cfg.M, T=cfg.T):
            truth_list.append(reports_truth)
            dev_list.append(reports_dev)

    out = evaluate_ic(
        cfg, values, truth_list, dev_list,
        samples=args.samples, seed0=args.seed,
        use_julia=args.use_julia
    )

    print("\n=== ε-IC result (time-monotone, mode=A) ===")
    print(f"Profiles evaluated : {len(truth_pairs)}")
    print(f"Average regret_mean: {out['regret_mean_avg']:.6f}")
    print(f"Worst-case regret  : {out['regret_max']:.6f}")

if __name__ == "__main__":
    main()
