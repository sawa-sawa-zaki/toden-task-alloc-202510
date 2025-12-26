
# RESTRICTED 戦略空間の生成と評価を担う

from typing import Dict, Tuple, List
from itertools import product

def generate_restricted_strategies(M: int, T: int):
    """
    RESTRICTED戦略：
    - 各 (m,t) に効用を与える
    - 辞退を0以下で表現
    - 単調性などは簡略化（本体コードと整合）
    """
    # 簡単化のため {-1, 0, 1} のみを使う
    values = [-1, 0, 1]
    strategies = []
    for v in product(values, repeat=M * T):
        strat = {}
        idx = 0
        for m in range(M):
            for t in range(T):
                strat[(m, t)] = v[idx]
                idx += 1
        strategies.append(strat)
    return strategies


def utility_of_assignment(true_pref: Dict[Tuple[int,int], float], alloc):
    """
    真の選好に基づき、割当から効用を計算
    """
    u = 0.0
    for (m, t) in alloc:
        u += true_pref.get((m, t), 0.0)
    return u
