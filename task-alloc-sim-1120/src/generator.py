import itertools
import numpy as np

MACHINE_RELS = ["A>B", "B>A", "A=B"]
UNACCEPTABLE = 99  # 辞退を表すランク値

def generate_consistent_with_truncation(num_machines=2, num_time=3):
    """
    Restricted: 一貫性（時間単調性）を満たし、かつアウトサイドオプション（辞退）を含む
    全てのユニークな選好タイプを生成する。
    """
    unique_prefs = set()
    results = []

    # 1. ベースとなる「完全な順序（辞退なし）」を生成
    base_candidates = []
    for ranks in itertools.product(range(6), repeat=6):
        rA1, rB1, rA2, rB2, rA3, rB3 = ranks
        
        # 時間単調性チェック
        if not (max(rA1, rB1) <= min(rA2, rB2)): continue
        if not (max(rA2, rB2) <= min(rA3, rB3)): continue
        
        valid_rel = False
        for rel1, rel2, rel3 in itertools.product(MACHINE_RELS, repeat=3):
            cond1 = (rel1=="A>B" and rA1<rB1) or (rel1=="B>A" and rB1<rA1) or (rel1=="A=B" and rA1==rB1)
            cond2 = (rel2=="A>B" and rA2<rB2) or (rel2=="B>A" and rB2<rA2) or (rel2=="A=B" and rA2==rB2)
            cond3 = (rel3=="A>B" and rA3<rB3) or (rel3=="B>A" and rB3<rA3) or (rel3=="A=B" and rA3==rB3)
            
            if cond1 and cond2 and cond3:
                valid_rel = True
                break
        
        if valid_rel:
            unique_vals = sorted(list(set(ranks)))
            rank_map = {v: i for i, v in enumerate(unique_vals)}
            norm_ranks = tuple(rank_map[r] for r in ranks)
            base_candidates.append(norm_ranks)

    # 2. 各ベース選好に対して「足切り」を適用
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
                results.append({
                    "ranks": new_ranks_tuple,
                    "relations": "DERIVED"
                })
                
    return results

def generate_all_permutations(num_machines=2, num_time=3):
    """Unrestricted (Strict): 厳密な全順列 (720通り)"""
    slots = []
    for t in range(num_time):
        for m in range(num_machines):
            slots.append((m, t))
    results = []
    for perm in itertools.permutations(slots):
        rank_map = {slot: r for r, slot in enumerate(perm)}
        ordered_ranks = []
        for t in range(num_time):
            for m in range(num_machines):
                ordered_ranks.append(rank_map[(m, t)])
        results.append({
            "ranks": tuple(ordered_ranks),
            "relations": "ANY"
        })
    return results

def generate_all_weak_orders(num_machines=2, num_time=3):
    """
    Unrestricted (Weak): 無差別を含む全弱順序 (46,833通り)
    """
    slots = []
    for t in range(num_time):
        for m in range(num_machines):
            slots.append((m, t))
            
    def get_partitions(collection):
        if len(collection) == 1:
            yield [collection]
            return
        first = collection[0]
        for smaller in get_partitions(collection[1:]):
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n+1:]
            yield [[first]] + smaller

    results = []
    for partition in get_partitions(slots):
        for ordered_partition in itertools.permutations(partition):
            rank_map = {}
            for rank, group in enumerate(ordered_partition):
                for slot in group:
                    rank_map[slot] = rank
            
            ordered_ranks = []
            for t in range(num_time):
                for m in range(num_machines):
                    ordered_ranks.append(rank_map[(m, t)])
            
            results.append({
                "ranks": tuple(ordered_ranks),
                "relations": "ANY"
            })
    return results

def ranks_to_matrix(ranks, base=10.0):
    """ランクを効用行列に変換"""
    arr = np.array(ranks)
    valid_mask = (arr != UNACCEPTABLE)
    
    if np.any(valid_mask):
        max_r = arr[valid_mask].max()
        scores = base ** (max_r - arr)
        scores[~valid_mask] = 0.0
    else:
        scores = np.zeros_like(arr, dtype=float)
    
    mat = np.zeros((2, 3))
    mat[0, 0] = scores[0]; mat[1, 0] = scores[1]
    mat[0, 1] = scores[2]; mat[1, 1] = scores[3]
    mat[0, 2] = scores[4]; mat[1, 2] = scores[5]
    return mat

def classify_lie_type(true_ranks, lie_ranks):
    """
    嘘のタイプを判定する
    Returns:
        "Type B": 構造変化 (順序逆転、または強選好の無差別化) -> 重い嘘
        "Type A": 厳格化 (無差別の順序付けのみ) -> 軽い嘘
        "None": 変化なし
    """
    n = len(true_ranks)
    is_type_b = False
    is_type_a = False
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            # 真の選好で i が j より好まれている
            if true_ranks[i] < true_ranks[j]:
                # 嘘で関係が崩れているか？ (逆転 or 無差別化)
                if lie_ranks[i] >= lie_ranks[j]:
                    # ただし、両方とも「辞退(99)」の場合は順序関係なしとみなす例外処理も必要だが
                    # 99 >= 99 は True になるので、辞退同士は「構造変化」とみなされてしまう可能性がある。
                    # -> 辞退同士は「変化なし」とみなすべき。
                    
                    if true_ranks[i] == UNACCEPTABLE and true_ranks[j] == UNACCEPTABLE:
                        continue
                    
                    return "Type B"
            
            # 真の選好で i と j が無差別
            elif true_ranks[i] == true_ranks[j]:
                # 嘘で順序がついた (i != j)
                if lie_ranks[i] != lie_ranks[j]:
                    is_type_a = True
                    
    if is_type_a:
        return "Type A"
    
    return "None"