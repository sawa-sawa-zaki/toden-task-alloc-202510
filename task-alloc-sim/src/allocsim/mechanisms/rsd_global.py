import numpy as np
from typing import Tuple, List, Dict, Any

def rsd_global_with_joint_caps(
    values_all: np.ndarray,    # (A, M, T)
    cap_t: np.ndarray,         # (T,)   時刻別（予測-バッファ）
    cap_m: np.ndarray,         # (M,)   マシン別（各tで共通）
    units_demand: np.ndarray,  # (A,)   エージェント総需要
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[int], List[Dict[str, Any]]]:
    """
    Serial Dictatorship (全時刻対象):
      - 優先順序 order に従い、各エージェントは自分の番で需要の残り分を
        可能な限りまとめて取得（1ユニットずつではなく「取り切る」）
      - 二層制約: 時刻 cap_t と マシン cap_m を同時に満たす
      - values_all[a,m,t] == -1 は割当禁止（アウトサイドオプション）
      - 選択過程を詳細ログ choice_log に記録
    """
    A, M, T = values_all.shape
    alloc = np.zeros((T, A, M), dtype=int)

    # 残余
    rem_a = units_demand.copy()          # 各agent 残需要
    rem_t = cap_t.copy()                 # 各時刻 残キャパ
    rem_m_t = np.tile(cap_m, (T, 1))     # (T, M) 各tのマシン残キャパ

    # 優先順序（固定）
    order = rng.permutation(A).tolist()

    # ログ
    choice_log: List[Dict[str, Any]] = []
    step = 0

    def best_feasible_item(a: int):
        """現時点で agent a が置ける (t,m) のうち価値最大のものを返す。無ければ (-1,-1,-inf)。"""
        best_t, best_m, best_v = -1, -1, -1e18
        for t in range(T):
            if rem_t[t] <= 0:
                continue
            for m in range(M):
                if rem_m_t[t, m] <= 0:
                    continue
                v = values_all[a, m, t]
                if v == -1:  # 禁止スロット
                    continue
                if v > best_v:
                    best_t, best_m, best_v = t, m, v
        return best_t, best_m, best_v

    # シリアル独裁：各エージェントが自分の番で「取り切る」
    for a in order:
        # もう置けない（需要0 or 全体キャパ0）ならスキップ
        if rem_a[a] <= 0 or rem_t.sum() <= 0:
            continue

        while rem_a[a] > 0 and rem_t.sum() > 0:
            t, m, v = best_feasible_item(a)
            if t == -1:
                # このエージェントはもう置ける場所がない
                break
            # 1ユニット割当
            alloc[t, a, m] += 1
            rem_a[a]   -= 1
            rem_t[t]   -= 1
            rem_m_t[t, m] -= 1

            # ログ
            choice_log.append({
                "step": step,
                "agent": int(a),
                "time": int(t),
                "machine": int(m),
                "value": float(v),
            })
            step += 1

    return alloc, order, choice_log
