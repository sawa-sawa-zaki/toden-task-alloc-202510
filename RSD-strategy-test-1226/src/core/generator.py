# src/core/generator.py
from __future__ import annotations

"""
generator（元zip task-alloc-sim-1120 互換）

- 真の選好（RESTRICTED）：時間単調性＋機械関係（A>B, B>A, A=B）＋truncation(辞退) を満たすタイプ
  -> M=2, T=3 で 216通りになるはず

- Type A 嘘：真の弱順序（無差別）を壊す「整合的な厳密化（refinement）」を全列挙
  -> ただし、実験側で「RESTRICTED集合に入るものだけ」を採用すれば 216^2 に閉じる
"""

import itertools
from typing import Dict, List, Tuple

MACHINE_RELS = ["A>B", "B>A", "A=B"]
UNACCEPTABLE = 99  # 辞退を表すランク値


def generate_consistent_with_truncation(num_machines: int = 2, num_time: int = 3):
    """
    元zip互換：Restricted（一貫性＋truncation）タイプをユニークに列挙する。
    返り値： [{"ranks": tuple(int,...), "relations": "DERIVED"}, ...]
    ranks の順序は (A,t1),(B,t1),(A,t2),(B,t2),(A,t3),(B,t3)（= 6要素）
    """
    assert num_machines == 2 and num_time == 3, "This generator is coded for M=2, T=3."

    unique_prefs = set()
    results = []

    # 1) ベースとなる「辞退なし」候補を列挙（6要素のrankを仮置き）
    base_candidates = []
    for ranks in itertools.product(range(6), repeat=6):
        rA1, rB1, rA2, rB2, rA3, rB3 = ranks

        # 時間単調性（t1のどれよりもt2が悪くならない・同値は可）
        if not (max(rA1, rB1) <= min(rA2, rB2)):
            continue
        if not (max(rA2, rB2) <= min(rA3, rB3)):
            continue

        # 各時点での機械関係（A>B, B>A, A=B）の整合性
        valid_rel = False
        for rel1, rel2, rel3 in itertools.product(MACHINE_RELS, repeat=3):
            cond1 = (rel1 == "A>B" and rA1 < rB1) or (rel1 == "B>A" and rB1 < rA1) or (rel1 == "A=B" and rA1 == rB1)
            cond2 = (rel2 == "A>B" and rA2 < rB2) or (rel2 == "B>A" and rB2 < rA2) or (rel2 == "A=B" and rA2 == rB2)
            cond3 = (rel3 == "A>B" and rA3 < rB3) or (rel3 == "B>A" and rB3 < rA3) or (rel3 == "A=B" and rA3 == rB3)
            if cond1 and cond2 and cond3:
                valid_rel = True
                break

        if valid_rel:
            # rank を 0..K-1 に正規化（同順位は同値のまま）
            unique_vals = sorted(list(set(ranks)))
            rank_map = {v: i for i, v in enumerate(unique_vals)}
            norm_ranks = tuple(rank_map[r] for r in ranks)
            base_candidates.append(norm_ranks)

    # 2) 各ベースに対して truncation（足切り）を入れる
    for base_ranks in base_candidates:
        distinct_ranks = sorted(list(set(base_ranks)))
        thresholds = distinct_ranks + [max(distinct_ranks) + 1]

        for th in thresholds:
            new_ranks = []
            for r in base_ranks:
                if r >= th:
                    new_ranks.append(UNACCEPTABLE)
                else:
                    new_ranks.append(r)

            new_ranks_tuple = tuple(new_ranks)
            if new_ranks_tuple not in unique_prefs:
                unique_prefs.add(new_ranks_tuple)
                results.append({"ranks": new_ranks_tuple, "relations": "DERIVED"})

    return results


def ranks_to_vals(ranks, num_machines=2, num_time=3):
    """
    効用仕様：
      - 辞退（99）: -1
      - 希望順位が良いほど {100000, 10000, 1000, 100, 10, 1}
    """
    assert num_machines == 2 and num_time == 3
    assert len(ranks) == 6

    value_levels = [1, 10, 100, 1000, 10000, 100000]

    # 辞退以外の rank
    valid_ranks = [r for r in ranks if r != 99]

    out = [[0.0 for _ in range(num_time)] for _ in range(num_machines)]

    # ★ 完全辞退タイプの特別処理
    if len(valid_ranks) == 0:
        idx = 0
        for t in range(num_time):
            for m in range(num_machines):
                out[m][t] = -1.0
                idx += 1
        return out

    max_rank = max(valid_ranks)

    idx = 0
    for t in range(num_time):
        for m in range(num_machines):
            r = ranks[idx]
            if r == 99:
                out[m][t] = -1.0
            else:
                out[m][t] = value_levels[max_rank - r]
            idx += 1

    return out



def generate_typeA_tie_break_lies(true_ranks: Tuple[int, ...], num_machines: int = 2, num_time: int = 3) -> List[Tuple[int, ...]]:
    """
    Type A 嘘：
    - UNACCEPTABLE を除く部分について、真の弱順序（同順位）を壊して厳密順位に“精緻化”する候補を列挙。
    - 真の厳密優劣（rankの大小関係）は絶対に変えない（同値の中だけ並べ替える）。

    返り値：精緻化した ranks（厳密順位で正規化したもの）
    """
    assert num_machines == 2 and num_time == 3 and len(true_ranks) == 6

    # acceptable positions
    positions = [i for i, r in enumerate(true_ranks) if r != UNACCEPTABLE]
    if not positions:
        return []

    # group by rank (ties)
    rank_to_pos: Dict[int, List[int]] = {}
    for i in positions:
        rank_to_pos.setdefault(true_ranks[i], []).append(i)

    # if no tie, no TypeA lies
    if all(len(ps) == 1 for ps in rank_to_pos.values()):
        return []

    # ranks in increasing order (better -> worse)
    distinct = sorted(rank_to_pos.keys())

    # for each tie-group, enumerate permutations of positions
    perm_lists = []
    for r in distinct:
        ps = rank_to_pos[r]
        if len(ps) == 1:
            perm_lists.append([tuple(ps)])
        else:
            perm_lists.append(list(itertools.permutations(ps)))

    lies = []
    # combine permutations across groups
    for choice in itertools.product(*perm_lists):
        # build strict ranking order over acceptable items
        ordered_positions = []
        for block in choice:
            ordered_positions.extend(block)

        # assign strict ranks 0..(K-1) according to this order
        new = [UNACCEPTABLE] * 6
        for new_rank, pos in enumerate(ordered_positions):
            new[pos] = new_rank

        lies.append(tuple(new))

    # 重複除去（理論上は不要だが保険）
    lies = list(dict.fromkeys(lies))
    return lies


def generate_all_permutations(num_machines: int = 2, num_time: int = 3):
    """（参考）Unrestricted strict: 6! = 720（今回TypeCを後回しなので実験では使わない）"""
    assert num_machines == 2 and num_time == 3
    slots = []
    for t in range(num_time):
        for m in range(num_machines):
            slots.append((m, t))

    results = []
    for perm in itertools.permutations(slots):
        rank_map = {slot: i for i, slot in enumerate(perm)}
        ordered_ranks = []
        for t in range(num_time):
            for m in range(num_machines):
                ordered_ranks.append(rank_map[(m, t)])
        results.append({"ranks": tuple(ordered_ranks)})
    return results
