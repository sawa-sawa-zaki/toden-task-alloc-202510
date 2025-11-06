# scripts/ic_check_global_truthA_devB.py
# 真の選好は A（機械順3×τ）で総当たり、嘘（報告）は B（無制約）で評価。
# 出力先は毎回ユニークなフォルダ。改善「比率」もCSVに保存。

import argparse
import os, sys, json, random
from datetime import datetime

# --- add src/ to import path ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
from allocsim.online_rsd_ttc import Config, evaluate_ic
from allocsim.strategy.time_monotone import (
    enumerate_truth_profiles_pair,      # Aドメイン（機械順×τ）の全組み合わせ（2人分）
)
from allocsim.strategy.unconstrained import (
    enumerate_devs_B_unconstrained,     # 無制約（弱順序）で嘘を生成（all/beamを選択）
)
from allocsim.report_from_values import (
    build_lexicographic_values,         # 値→レキシコ（桁優先）。効用スケール依存を避けるなら比率を併用
)

def make_unique_outdir(base="results", prefix="truthA_devB"):
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

def wrap_single_agent_devs(truth_reports, base_iter):
    """
    無制約Bのdevを『片側だけ嘘をつく』形にラップ。
    evaluate_ic() に渡せる (label, reports) を yield する。
    """
    for label, both in base_iter:
        # 左（agent 0）だけ嘘
        yield (dict(label, who=0), [both[0], truth_reports[1]])
        # 右（agent 1）だけ嘘
        yield (dict(label, who=1), [truth_reports[0], both[1]])

def main():
    ap = argparse.ArgumentParser()
    # モデル規模
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--q", type=int, nargs="+", default=[1, 1])
    ap.add_argument("--barS", type=int, nargs="+", default=[1, 1, 1])
    ap.add_argument("--buffer", type=int, nargs="+", default=[0, 1, 1])
    ap.add_argument("--window", type=int, default=2)
    # ショック
    ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    # 値（効用）のランク構造（デフォルトは「時間が早いほど嬉しい・A優先」）
    ap.add_argument("--rank_order", type=str,
                    default="(0,1),(1,1),(0,2),(1,2),(0,3),(1,3)")
    ap.add_argument("--ties", type=str, default="")
    # dev（嘘）の生成モード：全列挙 or ビーム（K件）
    ap.add_argument("--dev_mode", type=str, choices=["all", "beam"], default="beam")
    ap.add_argument("--dev_K", type=int, default=1000)
    ap.add_argument("--beam_width", type=int, default=64)
    ap.add_argument("--beam_iters", type=int, default=6)
    # 出力
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    # 出力先
    outdir = args.out_dir or make_unique_outdir()
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, "ic_truthA_devB.csv")

    # コンフィグ
    cfg = Config(
        A=args.A, M=args.M, T=args.T,
        units_demand=args.q,
        barS=args.barS,
        buffer=args.buffer,
        window=args.window,
        shock_p=args.p,
        shock_seed=args.seed
    )

    # 値（効用）：レキシコ（桁優先）で“時間早いほど嬉しい”を反映。
    ro = parse_rank_order(args.rank_order)
    ties = parse_ties(args.ties)
    Qmax = max(cfg.units_demand)
    values = build_lexicographic_values(
        A=cfg.A, M=cfg.M, T=cfg.T,
        rank_order=ro, Qmax=Qmax,
        ties=ties or None,
        agent_bias=[0.0] * cfg.A
    )

    # 真の選好：Aドメイン（機械順×τ）の2人直積（T=3 ⇒ 12×12=144ペア）
    rows = []
    pair_id = 0
    for label_pair, truth in enumerate_truth_profiles_pair(cfg.M, cfg.T):
        pair_id += 1

        # 嘘（無制約B）：全列挙 or ビーム
        base_iter = enumerate_devs_B_unconstrained(
            truth, cfg.M, cfg.T,
            mode=args.dev_mode,
            K=args.dev_K,
            beam_width=args.beam_width,
            iters=args.beam_iters,
            seed=args.seed
        )
        devs = wrap_single_agent_devs(truth, base_iter)

        # ε-IC評価
        out = evaluate_ic(
            cfg, values, truth, devs,
            samples=args.samples, seed0=args.seed, examples_per_agent=2
        )

        # 改善“比率”を追加（スケール非依存）
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
            "dev_mode": args.dev_mode,
            "dev_K": args.dev_K,
        }
        rows.append(row)

        if pair_id % 24 == 0:
            print(f"... {pair_id} / 144 done")

    # CSV保存
    import csv
    keys = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # サマリ
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

    print("\n=== Summary (Truth=A, Deviation=B) ===")
    print(f"Profiles evaluated        : {summary['profiles']}  (expected 144 for T=3)")
    print(f"Average regret_mean       : {summary['average_regret_mean']:.6f}")
    print(f"Average improve_ratio_mean: {summary['average_improve_ratio_mean']:.6f}")
    print(f"Worst-case pair_id        : {summary['worst_pair_id']}, "
          f"regret_max: {summary['worst_regret_max']:.6f}")
    print(f"CSV saved to              : {summary['csv_path']}")
    print(f"outdir                    : {outdir}")

if __name__ == "__main__":
    main()
