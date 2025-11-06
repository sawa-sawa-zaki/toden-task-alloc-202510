# scripts/ic_check_global_weak.py
import argparse
import os, sys, json, random
from datetime import datetime

# --- add src/ to sys.path ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
from allocsim.online_rsd_ttc import Config, evaluate_ic
from allocsim.strategy.weak_orders import (
    enumerate_truth_profiles_weak_pair,
)
from allocsim.strategy.time_monotone import (
    enumerate_devs_A_time_monotone,        # 既存：A=一貫性あり（時間単調＋機械順一貫＋OUT閾）
)
# 値は “効用そのもののスケールに依らない” 比率評価を使うため、単純スカラーで十分
# ここでは 1 をベースとし、報告は弱順序のみで効用は「取ったマスの個数×1」を加算。
# （効用比較は報告順序の影響を配分側が受けるため、それで十分）
def make_unit_values(A: int, M: int, T: int) -> np.ndarray:
    values = np.ones((A, M, T), dtype=float)
    return values

def make_unique_outdir(base="results", prefix="global_weak"):
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    rnd = random.randint(10000, 99999)
    outdir = os.path.join(base, f"{prefix}_{ts}_{rnd}")
    os.makedirs(outdir, exist_ok=True)
    return outdir

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
    ap.add_argument("--samples", type=int, default=150)   # 5,625ペアなので少し控えめでもOK
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    assert args.A == 2 and args.M == 2 and args.T == 2, "弱順序全列挙は A=2,M=2,T=2 を想定しています"

    outdir = args.out_dir or make_unique_outdir()
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, "ic_global_weak_A.csv")

    cfg = Config(
        A=args.A, M=args.M, T=args.T,
        units_demand=args.q,
        barS=args.barS,
        buffer=args.buffer,
        window=args.window,
        shock_p=args.p,
        shock_seed=args.seed
    )

    # 効用スケールの影響を消す：ユニット値（1）
    values = make_unit_values(cfg.A, cfg.M, cfg.T)

    rows = []
    pair_id = 0
    for label_pair, truth_reports in enumerate_truth_profiles_weak_pair(cfg.M, cfg.T):
        pair_id += 1
        # dev（嘘）は “一貫性あり(A)” の集合を使う（メッセージ制約Aの検証）
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
            "truth_p1_id": label_pair["p1_id"],
            "truth_p2_id": label_pair["p2_id"],
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

        if pair_id % 200 == 0:
            print(f"... {pair_id} / 5625 done")

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
        "outdir": outdir,
        "args": vars(args),
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary (Weak-orders x Weak-orders) ===")
    print(f"Profiles evaluated       : {summary['profiles']}  (expected 5625)")
    print(f"Average regret_mean      : {summary['average_regret_mean']:.6f}")
    print(f"Average improve_ratio_mean: {summary['average_improve_ratio_mean']:.6f}")
    print(f"Worst-case pair_id       : {summary['worst_pair_id']}, "
          f"regret_max: {summary['worst_regret_max']:.6f}")
    print(f"CSV saved to             : {summary['csv_path']}")
    print(f"outdir                   : {outdir}")

if __name__ == "__main__":
    main()
