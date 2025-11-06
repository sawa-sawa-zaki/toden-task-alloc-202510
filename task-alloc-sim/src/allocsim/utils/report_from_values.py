# src/allocsim/utils/report_from_values.py
from typing import List, Tuple, Union, Dict
import numpy as np

Item = Tuple[int, int]           # (m, t)
Level = List[Union[Item, str]]   # レベル集合（'OUT'含む）
Report = List[Level]             # 弱順序（レベルのリスト）
Reports = List[Report]

def make_reports_from_values(
    values_all: np.ndarray,
    out_last: bool = True,
) -> Reports:
    """
    各プレイヤー i について、values[i,m,t] を降順に並べ、同値は同レベルにまとめた弱順序Reportを返す。
    OUTは最後に付ける（out_last=True）。
    values_all: shape (A,M,T)
    """
    A, M, T = values_all.shape
    reports: Reports = []
    for i in range(A):
        cells = []
        for m in range(M):
            for t in range(T):
                # t は report側で 1-based にする（既存実装の前提に合わせる）
                cells.append((float(values_all[i, m, t]), m, t + 1))
        # 値が高い順にソート
        cells.sort(key=lambda x: x[0], reverse=True)
        # 同値でレベル分割
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
    レキシコグラフィック（桁優先）の値行列 values[i,m,t] を作る。
    - rank_order: 高い順に (m,t) を並べたリスト。例: [(0,1),(1,1),(0,2),(1,2)]
                  ※ t は 1-based 指定
    - Qmax: 需要上限（この個数を合計しても上位1個に勝たない基数設計）
    - ties: { (m,t): group_id }  同じ group_id を持つ項目は“同価”（同じ桁）にする
    - agent_bias: length A の微小バイアス（デフォルトNone）→ 0.0 推奨（完全対称）
    返り値: values shape (A,M,T)  ※ t は 0-basedに敷き詰める
    """
    assert len(rank_order) == M * T, "rank_order must list all M*T cells (m, t1-based)"
    if ties is None:
        ties = {}
    if agent_bias is None:
        agent_bias = [0.0] * A

    # 桁の基数：B = Qmax + 1 → 上位1個 > 下位(Qmax個)の合計
    B = Qmax + 1

    # rank_order は高い順で与えるので、順位rが小さいほど桁が大きい。
    # ties がある場合、同じ group_id の要素は同じ「代表順位」の桁を使う。
    # 内部では t を 0-based に直す。
    # exponent[(m, t0)] = B^k の k を入れる。
    exponent: Dict[Tuple[int, int], int] = {}
    # group の代表順位
    group_best_rank: Dict[int, int] = {}
    for r, (m, t1) in enumerate(rank_order):
        t0 = t1 - 1
        g = ties.get((m, t1))
        if g is not None:
            group_best_rank[g] = min(group_best_rank.get(g, r), r)
    for r, (m, t1) in enumerate(rank_order):
        t0 = t1 - 1
        g = ties.get((m, t1))
        rep_r = group_best_rank[g] if g is not None else r
        # 上から (M*T - 1) を降る（指数が高いほど価値が大）
        exponent[(m, t0)] = (M * T - 1) - rep_r

    values = np.zeros((A, M, T), dtype=float)
    for i in range(A):
        for m in range(M):
            for t0 in range(T):
                e = exponent[(m, t0)]
                values[i, m, t0] = (B ** e) + agent_bias[i]
    return values
