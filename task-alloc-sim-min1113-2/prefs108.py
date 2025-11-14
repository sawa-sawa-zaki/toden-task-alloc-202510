# prefs108.py
# 2計算機×3時間帯（6マス）上の「時間単調＋機械間の任意の好み＋無差別あり」
# を満たす弱順序をすべて列挙し、スコア行列に変換するユーティリティ

from itertools import product
import numpy as np

# 6マスのインデックスを固定で定義
# idx: 0   1   2   3   4   5
# cell: A1, B1, A2, B2, A3, B3
CELLS = [("A", 1), ("B", 1),
         ("A", 2), ("B", 2),
         ("A", 3), ("B", 3)]

MACHINE_RELS = ["A>B", "B>A", "A=B"]


def generate_time_monotone_prefs():
    """
    条件を満たす「6マス上の弱順序」をすべて列挙する。

    条件:
      - 各時間 t=1,2,3 での A_t/B_t 間の関係は A>B, B>A, A=B のいずれか
      - 時間単調性: 早い時間帯のスロットは、遅い時間帯のスロットより常に
        弱くは好まれる (無差別は許す)。
        つまり、任意の m1,m2 について t1 < t2 ⇒ (m1,t1) ≽ (m2,t2)
      - 無差別は時間内・時間間ともに許す（時間単調性を壊さない範囲で）

    戻り値: list[dict]
        各要素は {
            "ranks": (rA1, rB1, rA2, rB2, rA3, rB3),  # 0,1,2,... の整数ランク（小さいほど好ましい）
            "relations": (rel1, rel2, rel3),          # 各時間の機械間関係 ("A>B", "B>A", "A=B")
        }
    """
    prefs = set()   # canonical なキーで去重
    result = []

    for rel1, rel2, rel3 in product(MACHINE_RELS, repeat=3):
        # ranks は 6マスそれぞれに割り当てる整数ランク (0..5)
        for ranks in product(range(6), repeat=6):
            rA1, rB1, rA2, rB2, rA3, rB3 = ranks

            # --- 時間単調性のチェック ---
            # t1 < t2 のとき、任意のセルについて rank(t1) <= rank(t2)
            # ⇔ max_t1 <= min_t2 かつ max_t2 <= min_t3
            if not (max(rA1, rB1) <= min(rA2, rB2) and
                    max(rA2, rB2) <= min(rA3, rB3)):
                continue

            # --- 各時間帯での機械間関係のチェック ---
            ok = True
            for rel, (rA, rB) in [
                (rel1, (rA1, rB1)),
                (rel2, (rA2, rB2)),
                (rel3, (rA3, rB3)),
            ]:
                if rel == "A>B" and not (rA < rB):
                    ok = False
                    break
                if rel == "B>A" and not (rB < rA):
                    ok = False
                    break
                if rel == "A=B" and not (rA == rB):
                    ok = False
                    break
            if not ok:
                continue

            # --- ランク値の正規化（0..K-1 に詰める）---
            # 例: ranks=(2,2,5,5,5,5) → unique=[2,5] → map={2:0,5:1}
            #     → canon_ranks=(0,0,1,1,1,1)
            unique_sorted = sorted(set(ranks))
            mapping = {v: i for i, v in enumerate(unique_sorted)}
            canon_ranks = tuple(mapping[r] for r in ranks)

            key = canon_ranks + (rel1, rel2, rel3)
            if key in prefs:
                continue

            prefs.add(key)
            result.append({
                "ranks": canon_ranks,
                "relations": (rel1, rel2, rel3),
            })

    return result


def ranks_to_value_matrix(ranks, base=10.0):
    """
    ranks: 長さ6のタプル (rA1, rB1, rA2, rB2, rA3, rB3)
           0 が最も好ましいランク、値が大きいほど下位。
    base:  レキシコグラフィックの桁を作るための基数 (10 or 100 など)

    戻り値:
        values: shape (2,3) の numpy 配列
                values[0,t-1] = A_t のスコア
                values[1,t-1] = B_t のスコア
    """
    ranks = list(ranks)
    max_rank = max(ranks)
    K = max_rank + 1  # 使用しているランク段数

    # rank r に対して base^(K-1-r) を割り当てる（上位ほど大きい値）
    scores = [base ** (K - 1 - r) for r in ranks]

    values = np.zeros((2, 3), dtype=float)

    # インデックス対応:
    # 0:A1, 1:B1, 2:A2, 3:B2, 4:A3, 5:B3
    values[0, 0] = scores[0]  # A1
    values[1, 0] = scores[1]  # B1
    values[0, 1] = scores[2]  # A2
    values[1, 1] = scores[3]  # B2
    values[0, 2] = scores[4]  # A3
    values[1, 2] = scores[5]  # B3

    return values


if __name__ == "__main__":
    prefs = generate_time_monotone_prefs()
    print(f"Number of preference types (time-monotone, with ties): {len(prefs)}")
    # 期待される出力: 108
