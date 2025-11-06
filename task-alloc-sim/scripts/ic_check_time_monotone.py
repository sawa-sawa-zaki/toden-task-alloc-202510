# scripts/ic_check_time_monotone.py  ← 全文置換

import argparse
import os, sys, json, time, random
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from allocsim.report_from_values import (
    build_lexicographic_values,
    make_reports_from_values,
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
    enumerate_devs_A_time_monotone,   # ラベル付き版に置換済みであること
)
from allocsim.report_from_values import (
    build_lexicographic_values,
    make_reports_from_values,
)

def make_unique_outdir(base="results", prefix="run"):
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
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", type=str, default="A", choices=["A"])  # 今回はAのみ
    ap.add_argument("--out_dir", type=str, default="")
    # 値（効用）のランク仕様
    ap.add_argument("--rank_order", type=str, default="(0,1),(1,1),(0,2),(1,2)",
                    help="高価値→低価値の順に (m,t1-based) を並べる。例 '(0,1),(1,1),(0,2),(1,2)'")
    ap.add_argument("--ties", type=str, default="",
                    help="同価グループを '[(m,t1,g),(m,t1,g),...]' 形式で指定。例 '[(0,1,1),(1,2,1)]'")

    args = ap.parse_args()

    # 出力ディレクトリ（ユニーク）
    outdir = args.out_dir or make_unique_outdir()
    os.makedirs(outdir, exist_ok=True)

    cfg = Config(
        A=args.A, M=args.M, T=args.T,
        units_demand=args.q,
        barS=args.barS,
        buffer=args.buffer,
        window=args.window,
        shock_p=args.p,
        shock_seed=args.seed
    )

    # ---------- 値（効用）をレキシコ＋tiesで生成 ----------
    # rank_order のパース
    # 例: "(0,1),(1,1),(0,2),(1,2)" → [(0,1),(1,1),(0,2),(1,2)]
    ro = []
    for chunk in args.rank_order.split(")"):
        chunk = chunk.strip().strip(",")
        if not chunk:
            continue
        tup = chunk.strip().lstrip("(")
        m_str, t1_str = tup.split(",")
        ro.append((int(m_str), int(t1_str)))
    # ties のパース： "[(m,t1,g),...]" 形式 → dict {(m,t1): g}
    ties_dict = {}
    if args.ties.strip():
        s = args.ties.strip()
        s = s.strip().lstrip("[").rstrip("]")
        for part in s.split(")"):
            part = part.strip().strip(",")
            if not part:
                continue
            p2 = part.lstrip("(")
            m_str, t1_str, g_str = [x.strip() for x in p2.split(",")]
            ties_dict[(int(m_str), int(t1_str))] = int(g_str)

    Qmax = max(cfg.units_demand)
    values = build_lexicographic_values(
        A=cfg.A, M=cfg.M, T=cfg.T,
        rank_order=ro,           # 高→低の順
        Qmax=Qmax,               # 桁基数 B = Qmax+1
        ties=ties_dict or None,  # 同価指定
        agent_bias=[0.0]*cfg.A
    )

    # 真の選好（弱順序）＝ 値から自動生成（同価は同レベルへ）
    truth = make_reports_from_values(values)

    # ---------- dev（嘘）列挙（A: 一貫性あり） ----------
    devs = enumerate_devs_A_time_monotone(truth, cfg.M, cfg.T)  # (label, dev_reports) を yield

    # ---------- IC評価（平均EU・regret など） ----------
    out = evaluate_ic(cfg, values, truth, devs, samples=args.samples, seed0=args.seed, examples_per_agent=3)

    # 改善比率（relative）の計算： (EU_dev - EU_truth) / max(|EU_truth|, eps)
    eps = 1e-12
    EU_t = np.array(out["EU_truth"], dtype=float)
    EU_d = np.array(out["EU_best_dev"], dtype=float)
    rel = (EU_d - EU_t) / np.maximum(np.abs(EU_t), eps)
    out["improve_ratio_per_agent"] = rel.tolist()
    out["improve_ratio_mean"] = float(np.mean(rel))
    out["improve_ratio_max"] = float(np.max(rel))

    # 保存
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "out": out,
            "values": values.tolist()
        }, f, ensure_ascii=False, indent=2)

    # 表示
    print("=== ε-IC (A: consistent) ===")
    print(f"outdir            : {outdir}")
    print(f"A,M,T             : {cfg.A},{cfg.M},{cfg.T}")
    print(f"q                 : {cfg.units_demand}")
    print(f"barS              : {cfg.barS}")
    print(f"buffer            : {cfg.buffer}")
    print(f"window            : {cfg.window}")
    print(f"p                 : {cfg.shock_p}")
    print(f"samples           : {args.samples}")
    print(f"rank_order        : {args.rank_order}")
    print(f"ties              : {args.ties or 'None'}")
    print(f"regret_mean       : {out['regret_mean']:.6f}")
    print(f"regret_max        : {out['regret_max']:.6f}")
    print(f"improve_ratio_mean: {out['improve_ratio_mean']:.6f}")
    print(f"improve_ratio_max : {out['improve_ratio_max']:.6f}")
    print(f"EU_truth          : {out['EU_truth']}")
    print(f"EU_best_dev       : {out['EU_best_dev']}")
    print(f"best_dev_label    : {out.get('best_dev_label')}")
    print(f"num_dev_evaluated : {out['num_dev_evaluated']}")
    print("Examples (first up to 6):")
    for ex in (out.get("examples") or [])[:6]:
        print("  ", ex)

if __name__ == "__main__":
    main()
