from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import math

import numpy as np


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


def sample_shocks(cfg: Config, rng: random.Random) -> np.ndarray:
    """
    各時間帯 t についてショック δ_t をサンプル。
    - 確率 p_shock/2 で -1, p_shock/2 で +1, それ以外は 0
    （E[δ_t]=0 とは厳密には少しずれるが、簡略な仕様）
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

    values_report: shape (A,M,T)
        各エージェント a の報告効用 v[a,m,t]

    戻り値:
        alloc: shape (A, M, T), 各セル 0/1
    """
    A, M, T = cfg.A, cfg.M, cfg.T
    alloc = np.zeros((A, M, T), dtype=int)

    # 各時間帯の「RSDで使える容量」：期待供給 - バッファ
    residual_cap_t = np.array(cfg.barS, dtype=int) - np.array(cfg.buffer, dtype=int)
    residual_cap_t = np.maximum(residual_cap_t, 0)

    # 各エージェントの残り需要
    remaining_q = np.array(cfg.q, dtype=int)

    for t0 in range(T):
        # このステップで見える時間帯 [t0, t0+window-1]
        t_max = min(T - 1, t0 + cfg.window - 1)
        candidate_times = list(range(t0, t_max + 1))

        # RSD順をランダムに
        order = list(range(A))
        rng.shuffle(order)

        for a in order:
            while remaining_q[a] > 0:
                best_v = -math.inf
                best_m, best_t = None, None

                for t in candidate_times:
                    if residual_cap_t[t] <= 0:
                        continue
                    for m in range(M):
                        if alloc[a, m, t] > 0:
                            continue
                        v = values_report[a, m, t]
                        if v > best_v:
                            best_v = v
                            best_m, best_t = m, t

                if best_m is None:
                    # このタイムウィンドウ内でこれ以上取れるセルがない
                    break

                alloc[a, best_m, best_t] = 1
                remaining_q[a] -= 1
                residual_cap_t[best_t] -= 1

    return alloc


def _fill_positive_shock_for_t(
    cfg: Config,
    alloc: np.ndarray,
    values_true: np.ndarray,
    S: np.ndarray,
    t: int,
    rng: random.Random,
) -> None:
    """
    正のショックで S[t] が alloc[:,:,t].sum() より大きい場合に、
    残り需要のあるエージェントをランダムに繰り上げ割当する。
    """
    A, M, T = cfg.A, cfg.M, cfg.T

    while True:
        cur = int(alloc[:, :, t].sum())
        if cur >= S[t]:
            break

        # 各エージェントの現在の割当ユニット数
        alloc_units = alloc.reshape(A, -1).sum(axis=1)
        # まだ需要を満たしていないエージェント
        candidates_a = [a for a in range(A) if alloc_units[a] < cfg.q[a]]
        if not candidates_a:
            break

        # ランダムにエージェントを選ぶ
        a = rng.choice(candidates_a)

        # その時間 t で、まだ取っていない (m,t) のうち価値が最大のものを選ぶ
        best_v = -math.inf
        best_m = None
        for m in range(M):
            if alloc[a, m, t] > 0:
                continue
            v = values_true[a, m, t]
            if v > best_v:
                best_v = v
                best_m = m

        if best_m is None:
            # この時間帯で追加できるスロットがない
            break

        alloc[a, best_m, t] = 1
        # ループ先頭で cur と alloc_units は毎回取り直す


def _handle_negative_shock_for_t(
    cfg: Config,
    alloc: np.ndarray,
    values_true: np.ndarray,
    S: np.ndarray,
    t: int,
    rng: random.Random,
) -> None:
    """
    負のショックで S[t] < alloc[:,:,t].sum() の場合に、
    まず「無差別な好みを持つ範囲内で、タイムウィンドウ外(t' >= t+window)」へ
    追い出し移動を試み、それでも足りなければ価値の低いスロットから削除する。
    """
    A, M, T = cfg.A, cfg.M, cfg.T

    while True:
        cur = int(alloc[:, :, t].sum())
        if cur <= S[t]:
            break

        moved = False
        candidates_move = []

        # まず「無差別＆タイムウィンドウ外」に移せる候補を探す
        for a in range(A):
            for m in range(M):
                if alloc[a, m, t] <= 0:
                    continue
                v_here = values_true[a, m, t]
                # タイムウィンドウ外: t' >= t + window
                for t2 in range(t + cfg.window, T):
                    if S[t2] <= int(alloc[:, :, t2].sum()):
                        continue  # これ以上入れられない
                    if alloc[a, m, t2] > 0:
                        continue  # すでに持っている
                    if values_true[a, m, t2] == v_here:
                        # (a,m,t) を (a,m,t2) に移せる候補
                        candidates_move.append((a, m, t2))
                        break  # その (a,m) について最初に見つかった t2 で十分

        if candidates_move:
            a, m, t2 = rng.choice(candidates_move)
            alloc[a, m, t] = 0
            alloc[a, m, t2] = 1
            moved = True

        if moved:
            continue  # まだ cur > S[t] ならもう一度ループ

        # ここまで来たら、無差別移動では吸収しきれないので「価値の低い順に削除」
        worst_a, worst_m = None, None
        worst_v = math.inf
        for a in range(A):
            for m in range(M):
                if alloc[a, m, t] > 0:
                    v = values_true[a, m, t]
                    if v < worst_v:
                        worst_v = v
                        worst_a, worst_m = a, m

        if worst_a is None:
            break

        alloc[worst_a, worst_m, t] = 0
        # ループ先頭で cur を取り直す


def apply_shocks_and_ttc_like(
    cfg: Config,
    alloc: np.ndarray,
    values_true: np.ndarray,
    delta: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    """
    RSD の結果 alloc に対してショック δ_t を適用し、簡易な「追い出しTTCもどき」を行う。

    変更点:
      1. 正のショック (S_t > 現在の割当) のときは、残り需要のあるエージェントを
         ランダムに繰り上げて追加割当する。
      2. 負のショック (S_t < 現在の割当) のときは、まず「無差別かつタイムウィンドウ外」
         に移動できる割当を優先的に動かし、それでも足りない分のみ価値の低い順に削除。
    """
    A, M, T = cfg.A, cfg.M, cfg.T
    alloc_final = alloc.copy()

    # 各時間帯の実現供給 S_t
    S = np.zeros(T, dtype=int)
    for t in range(T):
        S_t = cfg.barS[t] + int(delta[t])
        if S_t < 0:
            S_t = 0
        S[t] = S_t

    # 各時間帯ごとに処理
    for t in range(T):
        # 1. 正のショック → ランダム繰り上げ
        if int(alloc_final[:, :, t].sum()) < S[t]:
            _fill_positive_shock_for_t(cfg, alloc_final, values_true, S, t, rng)

        # 2. 負のショック → 無差別移動 + 削除
        if int(alloc_final[:, :, t].sum()) > S[t]:
            _handle_negative_shock_for_t(cfg, alloc_final, values_true, S, t, rng)

    return alloc_final


def run_online_rsd_ttc(
    cfg: Config,
    values_true: np.ndarray,
    values_report: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1試行分のオンライン RSD+TTC もどきを回し、最終割当と真の効用を返す。

    values_true:   shape (A,M,T) 真の効用
    values_report: shape (A,M,T) 申告効用
    """
    # 報告に基づいて RSD
    alloc_rsd = run_rsd(cfg, values_report, rng)

    # ショックをサンプル
    delta = sample_shocks(cfg, rng)

    # ショック後の調整
    alloc_final = apply_shocks_and_ttc_like(cfg, alloc_rsd, values_true, delta, rng)

    # 真の効用で評価
    util = np.zeros(cfg.A, dtype=float)
    for a in range(cfg.A):
        util[a] = float(np.sum(values_true[a] * alloc_final[a]))

    return alloc_final, util
