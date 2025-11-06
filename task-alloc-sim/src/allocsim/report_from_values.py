# src/allocsim/report_from_values.py
from typing import List, Tuple, Union, Dict
import numpy as np

Item = Tuple[int, int]           # (m, t)
Level = List[Union[Item, str]]   # レベル集合（'OUT'含む）
Report = List[Level]             # 弱順序（レベルのリスト）
Reports = List[Report]

def make_reports_from_values(values_all: np.ndarray, out_last: bool = True) -> Reports:
    """
    values_all: shape (A,M,T)
    各プレイヤー i について、values[i,m,t] を降順に並べ、同値は同レベルにまとめた弱順序Reportを返す。
    OUTは最後に付ける（out_last=True）。
    """
    A, M, T = values_all.shape
    reports: Reports = []
    for i in range(A):
        cells = []
        for m in range(M):
            for t in range(T):
                cells.append((float(values_all[i, m, t]), m, t + 1))  # t は 1-based に
        cells.sort(key=lambda x: x[0], reverse=True)
        levels: Report = []
        cur_level: Level = []
        last_v = None
        for v, m, t1 in cells:
            if last_v is None or abs(v - last_v) < 1e-12:
                cur_level.append((m, t1))
            else:
                levels.append(cur_level)
                cur_level = [(m, t1)]
            last_v = v
        if cur_level:
            levels.append(cur_level)
        if out_last:
            levels.append(['OUT'])
        reports.append(levels)
    return reports

def build_lexicographic_values(
    A: int, M: int, T: int,
    rank_order: List[Tuple[int, int]],
    Qmax: int,
    ties: Dict[Tuple[int, int], int] = None,
    agent_bias: List[float] = None,
) -> np.ndarray:
    """
    レキシコ（桁優先）の値行列 values[i,m,t] を作る。
    - rank_order: 高い順に (m,t1-based)、長さ M*T
    - Qmax: 需要上限（上位1個 > 下位Qmax個の合計になる基数を作る）
    - ties: {(m,t1): group_id} 同じ group_id は同価（同じ指数）
    """
    assert len(rank_order) == M * T, "rank_order must list all M*T cells"
    if ties is None:
        ties = {}
    if agent_bias is None:
        agent_bias = [0.0] * A

    B = Qmax + 1  # 上位1個 > 下位Qmax個の合計
    exponent: Dict[Tuple[int, int], int] = {}
    # 同価グループの代表順位（最上位）を決める
    group_best_rank: Dict[int, int] = {}
    for r, (m, t1) in enumerate(rank_order):
        g = ties.get((m, t1))
        if g is not None:
            group_best_rank[g] = min(group_best_rank.get(g, r), r)
    for r, (m, t1) in enumerate(rank_order):
        t0 = t1 - 1
        g = ties.get((m, t1))
        rep_r = group_best_rank[g] if g is not None else r
        exponent[(m, t0)] = (M * T - 1) - rep_r

    values = np.zeros((A, M, T), dtype=float)
    for i in range(A):
        for m in range(M):
            for t0 in range(T):
                e = exponent[(m, t0)]
                values[i, m, t0] = (B ** e) + agent_bias[i]
    return values
