# scripts/ic_check_global.py  ← 全文置換

import argparse
import os, sys, json, random
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from datetime import datetime

from allocsim.report_from_values import (
    build_lexicographic_values,
)

# --- add src/ to sys.path ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
from allocsim.online_rsd_ttc import Config, evaluate_ic
from allocsim.strategy.time_monotone import (
    enumerate_truth_profiles_pair,        # 2人の真の選好ペア（τ×機械順）の全列挙
    enumerate_devs_A_time_monotone,       # dev（嘘）のラベル付き列挙（A:一貫性あり）
)
from allocsim.report_from_values import (
    build_lexicographic_values,
)

def make_unique_outdir(base="results", prefix="global"):
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    rnd = random.randint(10000, 99999)
    outdir = os.path.join(base, f"{prefix}_{ts}_{rnd}")
    os.makedirs(outdir, exist_ok=True)
    return outdir

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
    if not s.strip():
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
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--q", type=int, nargs="+", default=[1,1])
    ap.add_argument("--barS", type=int, nargs="+", default=[1,1])
    ap.add_argument("--buffer", type=int, nargs="+", default=[0,1])
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="")
    # 値（効用）のランク仕様
    ap.add_argument("--rank_order", type=str, default="(0,1),(1,1),(0,2),(1,2)")
    ap.add_argument("--ties", type=str, default="")
    args = ap.parse_args()

    outdir = args.out_dir or make_unique_outdir()
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, "ic_global_A.csv")

    cfg = Config(
        A=args.A, M=args.M, T=args.T,
        units_demand=args.q,
        barS=args.barS,
        buffer=args.buffer,
        window=args.window,
        shock_p=args.p,
        shock_seed=args.seed
    )

    # 値（効用）：桁優先＋ties可
    ro = parse_rank_order(args.rank_order)
    ties = parse_ties(args.ties)
    Qmax = max(cfg.units_demand)
    values = build_lexicographic_values(
        A=cfg.A, M=cfg.M, T=cfg.T,
        rank_order=ro,
        Qmax=Qmax,
        ties=ties or None,
        agent_bias=[0.0]*cfg.A
    )

    rows = []
    pair_id = 0
    for label_pair, truth_reports in enumerate_truth_profiles_pair(cfg.M, cfg.T):
        pair_id += 1
        devs = enumerate_devs_A_time_monotone(truth_reports, cfg.M, cfg.T)  # (label, dev_reports)

        out = evaluate_ic(
            cfg, values, truth_reports, devs,
            samples=args.samples, seed0=args.seed, examples_per_agent=2
        )

        # 改善比率（relative）
        eps = 1e-12
        EU_t = np.array(out["EU_truth"], dtype=float)
        EU_d = np.array(out["EU_best_dev"], dtype=float)
        rel = (EU_d - EU_t) / np.maximum(np.abs(EU_t), eps)

        row = {
            "pair_id": pair_id,
            "truth_p1_machine_order": label_pair["p1"]["machine_order"],
            "truth_p1_tau": label_pair["p1"]["tau"],
            "truth_p2_machine_order": label_pair["p2"]["machine_order"],
            "truth_p2_tau": label_pair["p2"]["tau"],
            "regret_mean": out["regret_mean"],
            "regret_max": out["regret_max"],
            "improve_ratio_mean": float(np.mean(rel)),
            "improve_ratio_max": float(np.max(rel)),
            "improve_ratio_per_agent": json.dumps(rel.tolist()),
            "EU_truth": json.dumps(out["EU_truth"]),
            "EU_best_dev": json.dumps(out["EU_best_dev"]),
            "best_dev_label": json.dumps(out.get("best_dev_label")),
            "examples": json.dumps(out.get("examples", [])),
        }
        rows.append(row)

        print(f"[{pair_id}] truth_p1={label_pair['p1']}, truth_p2={label_pair['p2']}  => "
              f"regret_max={out['regret_max']:.4f}, improve_ratio_max={row['improve_ratio_max']:.4f}, "
              f"best_dev_label={out.get('best_dev_label')}")

    # CSV保存
    import csv
    keys = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # サマリ保存
    avg_regret_mean = float(sum(r["regret_mean"] for r in rows) / len(rows)) if rows else 0.0
    avg_improve_ratio_mean = float(sum(r["improve_ratio_mean"] for r in rows) / len(rows)) if rows else 0.0
    worst = max(rows, key=lambda r: r["regret_max"]) if rows else None
    summary = {
        "profiles": len(rows),
        "average_regret_mean": avg_regret_mean,
        "average_improve_ratio_mean": avg_improve_ratio_mean,
        "worst_pair_id": worst["pair_id"] if worst else None,
        "worst_regret_max": worst["regret_max"] if worst else None,
        "csv_path": out_csv,
        "args": vars(args),
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"Profiles evaluated       : {summary['profiles']}")
    print(f"Average regret_mean      : {summary['average_regret_mean']:.6f}")
    print(f"Average improve_ratio_mean: {summary['average_improve_ratio_mean']:.6f}")
    print(f"Worst-case pair_id       : {summary['worst_pair_id']}, "
          f"regret_max: {summary['worst_regret_max']:.6f}")
    print(f"CSV saved to             : {summary['csv_path']}")
    print(f"outdir                   : {outdir}")

if __name__ == "__main__":
    main()
