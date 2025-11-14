#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小構成の RSD + (簡易)TTC + IC 検証コード

- モデルのセットアップ（Config）
- RSD とショック後の追い出し（TTC もどき）
- 真の選好タイプの列挙（時間単調 + 機械順 A>B / A=B / B>A）
- 「一貫性を満たした嘘」を許す / 許さないの切り替え
- IC の期待後悔（regret）を計算して CSV に保存

使い方:
    python ic_minimal.py

必要ライブラリ:
    numpy, pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import itertools
import csv
import os
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm



# ============================================================
# 1. モデル設定と選好生成
# ============================================================


@dataclass
class Config:
    A: int              # エージェント人数
    M: int              # 計算機台数
    T: int              # 時間帯数
    window: int         # タイムウィンドウの大きさ
    q: List[int]        # 各エージェントの需要（ユニット数）
    barS: List[int]     # 各時間帯 t の期待供給
    buffer: List[int]   # 各時間帯 t のバッファ
    p_shock: float      # ショック発生確率（±1）


# ---------- 選好タイプの定義 ----------
# 今は「時間が早いほど嬉しい」 + 「機械の順序が A>B/A=B/B>A の3パターン」のみ

MACHINE_ORDERS = ["A>B", "A=B", "B>A"]  # 機械順タイプの名前


def machine_sequence(order: str) -> List[List[int]]:
    """
    機械順の定義:
        "A>B":  [ [0], [1] ]
        "A=B":  [ [0,1] ]
        "B>A":  [ [1], [0] ]
    ここでは M=2 を想定。
    """
    if order == "A>B":
        return [[0], [1]]
    elif order == "B>A":
        return [[1], [0]]
    elif order == "A=B":
        return [[0, 1]]
    else:
        raise ValueError(f"unknown machine order: {order}")


def build_values_matrix(order: str, M: int, T: int) -> np.ndarray:
    """
    1人分の「真の効用行列」もしくは「報告された効用行列」を作る。

    - 時間が早いほど嬉しい（t=0 が最も上位）
    - 各 t 内での機械の順序は order で決まる
    - レキシコグラフィックに近いように、100,10,1,... の桁で値を与える
    - A=B のときは同じレベルにあるセルには同じ値を与える
    """
    seq = machine_sequence(order)
    # レベル（同値クラス）ごとにセルを列挙
    levels: List[List[Tuple[int, int]]] = []
    for t in range(T):
        for same_level in seq:
            cells = [(m, t) for m in same_level]
            levels.append(cells)

    K = len(levels)
    base = 100.0  # 時間優先を強くするため少し大きめ
    values = np.zeros((M, T), dtype=float)

    # 上位レベルほど大きい値
    rank = K - 1
    for cells in levels:
        v = base ** rank
        for (m, t) in cells:
            values[m, t] = v
        rank -= 1

    return values


# ============================================================
# 2. RSD とショック後の調整（簡易 TTC）
# ============================================================


def sample_shocks(cfg: Config, rng: random.Random) -> np.ndarray:
    """
    各時間帯 t についてショック δ_t をサンプル。
    - 確率 p_shock で ±1（符号は 1/2 の確率）
    - それ以外は 0
    （厳密には E[δ_t]=0 ではなくなりますが、ここでは簡略化）
    """
    delta = np.zeros(cfg.T, dtype=int)
    for t in range(cfg.T):
        u = rng.random()
        if u < cfg.p_shock / 2:
            delta[t] = -1
        elif u < cfg.p_shock:
            delta[t] = +1
        else:
            delta[t] = 0
    return delta


def run_rsd(cfg: Config, values_report: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    報告された効用行列にもとづき RSD を実行する。

    values_report: shape (A, M, T)
        - 各エージェント a の報告された価値 v[a,m,t]
    戻り値:
        alloc: shape (A, M, T), 各セルに 0/1 の割当
    """
    A, M, T = cfg.A, cfg.M, cfg.T
    alloc = np.zeros((A, M, T), dtype=int)

    # 各時間帯の「RSDで使える容量」：期待供給 - バッファ
    residual_cap_t = np.array(cfg.barS, dtype=int) - np.array(cfg.buffer, dtype=int)
    residual_cap_t = np.maximum(residual_cap_t, 0)

    # 各エージェントの残り需要
    remaining_q = np.array(cfg.q, dtype=int)

    for t0 in range(T):
        # このステップで使える時間帯の範囲 [t0, t0+window-1]
        t_max = min(T - 1, t0 + cfg.window - 1)
        candidate_times = list(range(t0, t_max + 1))

        # RSD 順をランダム化
        order = list(range(A))
        rng.shuffle(order)

        for a in order:
            while remaining_q[a] > 0:
                # 候補セルから最も価値の高いものを探す
                best_v = -math.inf
                best_m, best_t = None, None

                for t in candidate_times:
                    if residual_cap_t[t] <= 0:
                        continue
                    for m in range(M):
                        if alloc[a, m, t] > 0:
                            continue  # 1セル1ユニット前提
                        v = values_report[a, m, t]
                        if v > best_v:
                            best_v = v
                            best_m, best_t = m, t

                if best_m is None:
                    # これ以上割り当てられるセルがない
                    break

                alloc[a, best_m, best_t] = 1
                remaining_q[a] -= 1
                residual_cap_t[best_t] -= 1

    return alloc


def apply_shocks_and_ttc_like(
    cfg: Config,
    alloc: np.ndarray,
    values_true: np.ndarray,
    delta: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    """
    RSD の結果 alloc に対してショック δ_t を適用し、簡易な「追い出しTTC」を行う。

    - 各時間帯 t で実現供給 S_t = barS_t + δ_t
    - alloc[:, :, t].sum() > S_t なら、価値が低い方から追い出す（OUTSIDEへ）
    - 実現供給が増えたとき（S_t が大きい）は、需要が固定なので増やさない（余っても放置）

    values_true: shape (A,M,T)  真の効用（追い出し対象の優先順位に使う）

    戻り値:
        alloc_final: shape (A,M,T)
    """
    A, M, T = cfg.A, cfg.M, cfg.T
    alloc_final = alloc.copy()

    for t in range(T):
        S_t = cfg.barS[t] + int(delta[t])
        if S_t < 0:
            S_t = 0

        cur = int(alloc_final[:, :, t].sum())
        # 供給減少（追い出しが必要）
        if cur > S_t:
            drop = cur - S_t
            for _ in range(drop):
                worst_a, worst_m = None, None
                worst_v = math.inf
                for a in range(A):
                    for m in range(M):
                        if alloc_final[a, m, t] > 0:
                            v = values_true[a, m, t]
                            if v < worst_v:
                                worst_v = v
                                worst_a, worst_m = a, m
                if worst_a is None:
                    break
                alloc_final[worst_a, worst_m, t] -= 1

    return alloc_final


def run_online_rsd_ttc(
    cfg: Config,
    values_true: np.ndarray,
    values_report: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1回の試行（真の効用 values_true、報告 values_report 固定）について、
    RSD → ショック → 追い出しTTC を実行し、最終割当と効用を返す。

    戻り値:
        alloc_final: shape (A,M,T)
        util:        shape (A,)
    """
    # RSD は報告された値に基づいて行われる
    alloc_rsd = run_rsd(cfg, values_report, rng)

    # ショックをサンプル
    delta = sample_shocks(cfg, rng)

    # ショック後に追い出し（簡易 TTC）
    alloc_final = apply_shocks_and_ttc_like(cfg, alloc_rsd, values_true, delta, rng)

    # 真の効用で評価
    util = np.zeros(cfg.A, dtype=float)
    for a in range(cfg.A):
        util[a] = float(np.sum(values_true[a] * alloc_final[a]))

    return alloc_final, util


# ============================================================
# 3. IC（耐戦略性）の検証
# ============================================================


def enumerate_truth_profiles(A: int) -> List[Tuple[str, ...]]:
    """
    2人ゲーム用：
    各プレイヤーに MACHINE_ORDERS から1つずつ割り当てた真のタイプの全組み合わせ。
    A=2 の場合、3^2 = 9 通り。
    """
    return list(itertools.product(MACHINE_ORDERS, repeat=A))


def values_from_types(types: Tuple[str, ...], M: int, T: int) -> np.ndarray:
    """
    types[a] に対応する真の価値行列をまとめて返す。
    戻り値 shape (A,M,T)
    """
    A = len(types)
    vals = np.zeros((A, M, T), dtype=float)
    for a, order in enumerate(types):
        vals[a] = build_values_matrix(order, M, T)
    return vals


def report_values_from_types(types: Tuple[str, ...], M: int, T: int) -> np.ndarray:
    """
    報告行列も同じロジックで生成する（types が報告タイプの場合）。
    """
    return values_from_types(types, M, T)


def dev_type_candidates(
    true_type: str,
    allow_consistent_lies: bool,
) -> List[str]:
    """
    1人分の dev 候補タイプを返す。

    - allow_consistent_lies=True の場合：
        MACHINE_ORDERS 全部（真実型を含む）
    - False の場合：
        [true_type] のみ（つまり dev なし）
    """
    if allow_consistent_lies:
        return list(MACHINE_ORDERS)
    else:
        return [true_type]


def evaluate_ic_for_profile(
    cfg: Config,
    truth_types: Tuple[str, ...],
    samples: int,
    seed0: int,
    allow_consistent_lies: bool,
) -> Dict[str, Any]:
    """
    固定された真の選好プロファイル truth_types に対し、
    「一貫性を満たした嘘」を許す場合の IC を評価する。

    - 各プレイヤー i について：
        dev_type ∈ dev_type_candidates(true_type_i) を列挙
        （真実タイプも含むので、最良の dev_type が真実なら regret=0）
    - 各 dev_type ごとに Monte Carlo で samples 回回し、
        regret_mean[i, dev_type] = E[ u_i(dev) - u_i(truth) ]
      を計算。
    """
    A, M, T = cfg.A, cfg.M, cfg.T

    # 真の効用行列
    values_true = values_from_types(truth_types, M, T)

    # あらかじめ真実申告時の EU を計算しておく
    EU_truth = np.zeros(A, dtype=float)

    rng = random.Random(seed0)
    for s in range(samples):
        # 毎回同じ真の効用・真実報告
        values_report_truth = report_values_from_types(truth_types, M, T)
        _, util_truth = run_online_rsd_ttc(cfg, values_true, values_report_truth, rng)
        EU_truth += util_truth

    EU_truth /= samples

    # dev の探索
    best_regret_mean = np.zeros(A, dtype=float)
    best_dev_type: List[str] = [truth_types[a] for a in range(A)]

    for i in range(A):  # dev するプレイヤー
        true_t = truth_types[i]
        cand_types = dev_type_candidates(true_t, allow_consistent_lies)

        for dev_t in cand_types:
            # dev_t == true_t のケースも含める（regret=0基準）
            EU_dev = 0.0
            rng_dev = random.Random(seed0 + 1000 * (i + 1) + hash(dev_t) % 1000)

            dev_types = list(truth_types)
            dev_types[i] = dev_t
            dev_types_tuple = tuple(dev_types)

            for s in range(samples):
                values_report_dev = report_values_from_types(dev_types_tuple, M, T)
                _, util_dev = run_online_rsd_ttc(cfg, values_true, values_report_dev, rng_dev)
                EU_dev += util_dev[i]

            EU_dev /= samples
            regret_mean = EU_dev - EU_truth[i]

            if regret_mean > best_regret_mean[i] + 1e-9:
                best_regret_mean[i] = regret_mean
                best_dev_type[i] = dev_t

    result = {
        "truth_types": truth_types,
        "EU_truth": EU_truth.tolist(),
        "best_regret_mean": best_regret_mean.tolist(),
        "best_dev_type": best_dev_type,
    }
    return result


def run_ic_experiment(
    cfg: Config,
    samples: int = 20,
    seed0: int = 0,
    allow_consistent_lies: bool = True,
    out_csv: str = "results_ic_minimal.csv",
) -> pd.DataFrame:
    """
    全ての真の選好プロファイルについて IC を評価し、CSV に保存しつつ DataFrame も返す。
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    truth_profiles = enumerate_truth_profiles(cfg.A)
    rows = []


    print(f"[INFO] Evaluating {len(truth_profiles)} profiles...")

    for pair_id, truth_types in enumerate(
        tqdm(truth_profiles, desc="IC profiles", ncols=80)
    ):
        out = evaluate_ic_for_profile(
            cfg=cfg,
            truth_types=truth_types,
            samples=samples,
            seed0=seed0 + pair_id * 10000,
            allow_consistent_lies=allow_consistent_lies,
        )

        EU_truth = out["EU_truth"]
        best_regret_mean = out["best_regret_mean"]
        best_dev_type = out["best_dev_type"]

        row = {
            "pair_id": pair_id,
            "truth_p1": out["truth_types"][0],
            "truth_p2": out["truth_types"][1] if cfg.A >= 2 else None,
            "EU_truth_p1": EU_truth[0],
            "EU_truth_p2": EU_truth[1] if cfg.A >= 2 else None,
            "best_regret_p1": best_regret_mean[0],
            "best_regret_p2": best_regret_mean[1] if cfg.A >= 2 else None,
            "best_dev_type_p1": best_dev_type[0],
            "best_dev_type_p2": best_dev_type[1] if cfg.A >= 2 else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # コンソールに簡易サマリを出力
    print("=== IC summary (minimal) ===")
    print(f"A={cfg.A}, M={cfg.M}, T={cfg.T}, window={cfg.window}")
    print(f"q={cfg.q}, barS={cfg.barS}, buffer={cfg.buffer}, p_shock={cfg.p_shock}")
    print(f"samples per profile: {samples}")
    print(f"allow_consistent_lies: {allow_consistent_lies}")
    print(f"CSV saved to: {out_csv}")

    # ワーストケースを表示
    for i in range(cfg.A):
        col = f"best_regret_p{i+1}"
        idx = df[col].idxmax()
        print(f"  Player {i+1}: worst-case regret_mean = {df.loc[idx, col]:.4f} at pair_id={df.loc[idx, 'pair_id']}")

    return df


# ============================================================
# 4. メイン（ここをいじれば実験条件を変えられる）
# ============================================================


def main():
    # ---- モデル設定 ----
    cfg = Config(
        A=2,              # プレイヤー数
        M=2,              # 計算機数
        T=3,              # 時間帯数
        window=2,         # タイムウィンドウの大きさ
        q=[1, 1],         # 各プレイヤーの需要
        barS=[1, 1, 1],   # 各時間帯の期待供給
        buffer=[0, 0, 0], # 各時間帯のバッファ
        p_shock=0.5,      # ショックの発生確率
    )

    # ---- IC 実験を実行 ----
    df = run_ic_experiment(
        cfg=cfg,
        samples=20,                 # 各プロファイルあたりの試行回数
        seed0=0,
        allow_consistent_lies=True,  # True: 一貫性のある嘘を許す（機械順だけ変える）
        out_csv="results_ic_minimal/ic_results_A2M2T3.csv",
    )

    # 簡易可視化（テキスト）：どのタイプ組み合わせが危ないか
    print("\nTop 5 profiles by Player1 regret:")
    print(
        df.sort_values("best_regret_p1", ascending=False)
          .head(5)[["pair_id", "truth_p1", "truth_p2", "best_regret_p1", "best_dev_type_p1"]]
          .to_string(index=False)
    )


if __name__ == "__main__":
    main()
