import itertools
import numpy as np

MACHINE_RELS = ["A>B", "B>A", "A=B"]

def generate_time_monotone_prefs(num_machines=2, num_time=3):
    results = []
    seen = set()
    for ranks in itertools.product(range(6), repeat=6):
        rA1, rB1, rA2, rB2, rA3, rB3 = ranks
        if not (max(rA1, rB1) <= min(rA2, rB2)): continue
        if not (max(rA2, rB2) <= min(rA3, rB3)): continue
        valid_rel_combo = False
        for rel1, rel2, rel3 in itertools.product(MACHINE_RELS, repeat=3):
            if rel1 == "A>B" and not (rA1 < rB1): continue
            if rel1 == "B>A" and not (rB1 < rA1): continue
            if rel1 == "A=B" and not (rA1 == rB1): continue
            if rel2 == "A>B" and not (rA2 < rB2): continue
            if rel2 == "B>A" and not (rB2 < rA2): continue
            if rel2 == "A=B" and not (rA2 == rB2): continue
            if rel3 == "A>B" and not (rA3 < rB3): continue
            if rel3 == "B>A" and not (rB3 < rA3): continue
            if rel3 == "A=B" and not (rA3 == rB3): continue
            current_rels = (rel1, rel2, rel3)
            valid_rel_combo = True
            break 
        if not valid_rel_combo: continue
        unique_vals = sorted(list(set(ranks)))
        rank_map = {v: i for i, v in enumerate(unique_vals)}
        norm_ranks = tuple(rank_map[r] for r in ranks)
        key = (norm_ranks, current_rels)
        if key not in seen:
            seen.add(key)
            results.append({"ranks": norm_ranks, "relations": current_rels})
    return results

def ranks_to_matrix(ranks, base=10.0):
    arr = np.array(ranks)
    max_r = arr.max()
    scores = base ** (max_r - arr)
    mat = np.zeros((2, 3))
    mat[0, 0] = scores[0]; mat[1, 0] = scores[1]
    mat[0, 1] = scores[2]; mat[1, 1] = scores[3]
    mat[0, 2] = scores[4]; mat[1, 2] = scores[5]
    return mat