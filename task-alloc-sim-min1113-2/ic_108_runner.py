# ic_108_runner.py
# 108種類の選好を用いて、2人市場における IC を検証するメインスクリプト
#
# やること:
# 1. 2人-2計算機-3時間帯の市場設定(Config)
# 2. prefs108.generate_time_monotone_prefs() で 108種類の選好を生成し、
#    各タイプに ranks_to_value_matrix を対応づける
# 3. 「真の選好ペア」(type_i, type_j) をいくつか選び（全部でも良いが計算量注意）、
#    各プレイヤーが 108種類すべての他タイプで嘘をついたときの期待効用差を評価
# 4. IC違反（regret>0）のケースを violations CSV に書き出し、
#    各プロファイルの最大 regret を summary CSV に書き出す

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import itertools
import os
import random
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from prefs108 import generate_time_monotone_prefs, ranks_to_value_matrix
from mechanism_rsd_ttc import Config, run_online_rsd_ttc


@dataclass
class ICConfig:
    samples: int = 20             # 各 (truth, dev) 組み合わせあたりの試行数
    seed0: int = 0                 # 乱数のベースシード
    max_profiles: int | None = 200 # 真プロファイルの最大数(Noneなら全通り108^2)


def build_values_for_types(
    type_indices: Tuple[int, int],
    pref_list: List[dict],
    base: float = 10.0,
) -> np.ndarray:
    """
    type_indices: (idx1, idx2) それぞれ 0..len(pref_list)-1
    戻り値: values_true shape (2,2,3)
    """
    vals = np.zeros((2, 2, 3), dtype=float)
    for a, idx in enumerate(type_indices):
        ranks = pref_list[idx]["ranks"]
        vals[a] = ranks_to_value_matrix(ranks, base=base)
    return vals


def evaluate_profile_ic(
    cfg: Config,
    ic_cfg: ICConfig,
    pref_list: List[dict],
    type_indices: Tuple[int, int],
) -> Tuple[list, list]:
    """
    1つの真の選好プロファイル (type_i, type_j) について、
    各プレイヤー a=0,1 が全108タイプで嘘をつくケースを評価する。

    戻り値:
      - summary_rows: 各プレイヤーについての「最大regret」情報（長さ2）
      - violation_rows: IC違反 (regret>0) が出た全ケースの詳細リスト
    """
    A = 2
    all_type_ids = list(range(len(pref_list)))

    # 真の効用行列
    values_true = build_values_for_types(type_indices, pref_list, base=10.0)

    # 共通乱数用 seed 列（真実・各devで共有）
    seeds = [ic_cfg.seed0 + s for s in range(ic_cfg.samples)]

    # --- 真実申告の期待効用 ---
    EU_truth = np.zeros(A, dtype=float)

    for s, seed in enumerate(seeds):
        rng = random.Random(seed)
        # 真実申告: report = values_true
        _, util_truth = run_online_rsd_ttc(cfg, values_true, values_true, rng)
        EU_truth += util_truth
    EU_truth /= ic_cfg.samples

    # --- dev ケース ---
    best_regret = np.zeros(A, dtype=float)
    best_dev_type = [-1 for _ in range(A)]

    violation_rows = []

    for a in range(A):  # dev するプレイヤー
        for dev_idx in all_type_ids:
            # 「嘘をつかない」ケースも一応評価する（regret=0 が基準）
            # ただし dev_idx == true_idx で他と同じ処理
            dev_type_indices = list(type_indices)
            dev_type_indices[a] = dev_idx

            # dev時の報告効用
            values_report = build_values_for_types(tuple(dev_type_indices), pref_list, base=10.0)

            # 期待効用 (共通乱数)
            EU_dev_a = 0.0
            for s, seed in enumerate(seeds):
                rng = random.Random(seed)
                _, util_dev = run_online_rsd_ttc(cfg, values_true, values_report, rng)
                EU_dev_a += util_dev[a]
            EU_dev_a /= ic_cfg.samples

            regret = EU_dev_a - EU_truth[a]

            # IC違反 (regret > 0) を記録
            if regret > 1e-9:
                violation_rows.append({
                    "truth_type_p1": type_indices[0],
                    "truth_type_p2": type_indices[1],
                    "player": a + 1,
                    "dev_type": dev_idx,
                    "EU_truth": EU_truth[a],
                    "EU_dev": EU_dev_a,
                    "regret": regret,
                })

            # プロファイル内での最大regretを更新
            if regret > best_regret[a]:
                best_regret[a] = regret
                best_dev_type[a] = dev_idx

    # summary 行（プレイヤーごと）
    summary_rows = []
    for a in range(A):
        summary_rows.append({
            "truth_type_p1": type_indices[0],
            "truth_type_p2": type_indices[1],
            "player": a + 1,
            "EU_truth": EU_truth[a],
            "max_regret": best_regret[a],
            "best_dev_type": best_dev_type[a],
        })

    return summary_rows, violation_rows


def run_ic_experiment(
    cfg: Config,
    ic_cfg: ICConfig,
    out_dir: str = "results_ic_108",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 108種類の選好を生成
    prefs = generate_time_monotone_prefs()
    n_types = len(prefs)
    print(f"[INFO] preference types generated: {n_types}")  # 期待される: 108

    # 真の選好ペア（2人）すべての組み合わせ
    all_profiles = list(itertools.product(range(n_types), repeat=2))
    total_profiles = len(all_profiles)  # 108^2 = 11664
    print(f"[INFO] total truth profiles: {total_profiles}")

    # 計算量がやばいので、必要ならサンプルに制限
    if ic_cfg.max_profiles is not None and ic_cfg.max_profiles < total_profiles:
        rng = random.Random(ic_cfg.seed0)
        profiles = rng.sample(all_profiles, ic_cfg.max_profiles)
        print(f"[INFO] sampling {ic_cfg.max_profiles} profiles out of {total_profiles}")
    else:
        profiles = all_profiles
        print(f"[INFO] using all profiles")

    summary_rows_all = []
    violation_rows_all = []

    for type_indices in tqdm(profiles, desc="IC profiles", ncols=80):
        summary_rows, violation_rows = evaluate_profile_ic(cfg, ic_cfg, prefs, type_indices)
        summary_rows_all.extend(summary_rows)
        violation_rows_all.extend(violation_rows)

    # DataFrame にして CSV 出力
    df_summary = pd.DataFrame(summary_rows_all)
    df_violation = pd.DataFrame(violation_rows_all)

    summary_path = os.path.join(out_dir, "ic_summary.csv")
    violation_path = os.path.join(out_dir, "ic_violations.csv")

    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    df_violation.to_csv(violation_path, index=False, encoding="utf-8-sig")

    print("\n=== IC experiment done ===")
    print(f"summary   CSV: {summary_path}")
    print(f"violations CSV: {violation_path}")
    print(f"profiles evaluated: {len(profiles)}")
    print(f"violations found : {len(df_violation)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=200,
                   help="Monte Carlo samples per (truth,dev) pair")
    p.add_argument("--seed", type=int, default=0,
                   help="base random seed")
    p.add_argument("--max_profiles", type=int, default=200,
                   help="max number of truth profiles to evaluate (None for all)")
    p.add_argument("--out_dir", type=str, default="results_ic_108",
                   help="output directory for CSVs")
    return p.parse_args()


def main():
    args = parse_args()

    # 市場の設定（ここが基本の 2-2-3 モデル）
    cfg = Config(
        A=2,              # プレイヤー数
        M=2,              # 計算機数
        T=3,              # 時間帯数
        window=2,         # タイムウィンドウの大きさ
        q=[2, 2],         # 各プレイヤーの需要 (1ユニットずつ)
        barS=[1, 1, 1],   # 各時間帯の期待供給
        buffer=[1, 1, 1], # 各時間帯のバッファ
        p_shock=0.25,      # ショック発生確率
    )

    ic_cfg = ICConfig(
        samples=args.samples,
        seed0=args.seed,
        max_profiles=None if args.max_profiles < 0 else args.max_profiles,
    )

    run_ic_experiment(cfg, ic_cfg, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
