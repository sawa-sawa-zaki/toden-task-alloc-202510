
# deviations.py
from typing import List, Tuple, Union
Item = Tuple[int, int]  # (m, t)
Level = List[Union[Item, str]]  # 'OUT' を含む

def generate_truth_time_monotone(
    A: int, M: int, T: int,
    machine_bias: List[int]  # 長さA、+1: A≻B, 0: A∼B, -1: B≻A
) -> List[List[Level]]:
    """
    各プレイヤー i の真の弱順序（レベル集合のリスト）を生成。
    時間は必ず 1 ≻ 2 ≻ ... ≻ T。
    機械順は全時間で一貫（A≻B / A∼B / B≻A）。
    最後に OUT。
    返り値の型: [player i][level idx] = [(m,t), ...] or 'OUT'
    """
    prefs = []
    for i in range(A):
        levels: List[Level] = []
        for t in range(1, T+1):
            if machine_bias[i] > 0:      # A≻B
                level = [(0, t), (1, t)] if M >= 2 else [(0, t)]
            elif machine_bias[i] < 0:    # B≻A
                level = [(1, t), (0, t)] if M >= 2 else [(0, t)]
            else:                        # A∼B（無差別）
                level = [(0, t)]
                if M >= 2: level.append((1, t))
            levels.append(level)
        levels.append(['OUT'])
        prefs.append(levels)
    return prefs


def enumerate_devs_A_time_monotone(
    truth_levels: List[List[Level]],  # 全プレイヤー分
    M: int, T: int,
    allow_tie: bool = True
):
    """
    A: 一貫性ありの dev を全列挙。
    - 時間は 1≻2≻...≻T を維持
    - 機械順は全時間で一定（A≻B, A∼B, B≻A）
    - OUT カットオフ τ を入れる（τ in 0..T）
    """
    Agt = len(truth_levels)
    devs = []

    machine_orders = []
    if M == 1:
        machine_orders = [[0]]
    else:
        machine_orders = [[0,1]]          # A≻B
        if allow_tie: machine_orders.append([0,0])  # 疑似的に同率扱い（実装側で同レベルに置く）
        machine_orders.append([1,0])      # B≻A

    for tau in range(0, T+1):  # τ=0..T
        for mo in machine_orders:
            proposal = []
            for i in range(Agt):
                levels: List[Level] = []
                # t=1..T の順は固定
                for t in range(1, T+1):
                    if tau >= 1 and t > tau:
                        continue  # ここはOUTより下に落とす
                    if M == 1:
                        levels.append([(0,t)])
                    else:
                        if mo == [0,0] and allow_tie:
                            levels.append([(0,t),(1,t)])  # A∼B
                        elif mo == [0,1]:
                            levels.append([(0,t),(1,t)])  # A≻B
                        else:
                            levels.append([(1,t),(0,t)])  # B≻A
                levels.append(['OUT'])
                # tau < T のとき、 (t > tau) の項目は OUT の後ろ扱いになる（実装側で guard）
                proposal.append(levels)
            devs.append(proposal)
    return devs
