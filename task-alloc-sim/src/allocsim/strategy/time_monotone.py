# src/allocsim/strategy/time_monotone.py
from typing import List, Tuple, Union, Iterable
import itertools
import random

Item = Tuple[int, int]                   # (m, t)
Level = List[Union[Item, str]]           # レベル集合。'OUT' を含み得る
Report = List[Level]                     # 弱順序（レベルのリスト）
Reports = List[Report]                   # プレイヤーごとのリスト

def generate_truth_time_monotone(
    A: int, M: int, T: int,
    machine_bias: List[int],   # len=A, +1: A≻B, 0: A∼B(無差別), -1: B≻A
) -> Reports:
    """
    各プレイヤー i の真の弱順序（時間単調：1≻2≻...≻T）を生成。
    - 全時間で機械順は一貫（A≻B / A∼B / B≻A）。
    - 最後に OUT を追加。
    """
    assert len(machine_bias) == A
    prefs: Reports = []
    for i in range(A):
        mb = machine_bias[i]
        levels: Report = []
        for t in range(1, T+1):
            if M == 1:
                levels.append([(0, t)])
            else:
                if mb > 0:        # A≻B
                    levels.append([(0, t), (1, t)])
                elif mb < 0:      # B≻A
                    levels.append([(1, t), (0, t)])
                else:             # A∼B
                    levels.append([(0, t), (1, t)])  # 同じレベルで無差別
        levels.append(['OUT'])
        prefs.append(levels)
    return prefs

def enumerate_devs_A_time_monotone(truth: Reports, M: int, T: int):
    """
    A: 一貫性ありの dev を列挙（ラベル付きで yield）。
    yield: (label_dict, dev_reports)
      label_dict = {"regime":"A", "tau": τ, "machine_order": "A>B"|"A=B"|"B>A"}
    """
    Agt = len(truth)
    if M == 1:
        machine_orders = [("A",)]  # 名目
    else:
        machine_orders = [("A>B",), ("A=B",), ("B>A",)]

    for tau in range(0, T+1):        # τ: OUT を挿入する閾値（τ=0 は OUT 最下位）
        for mo in machine_orders:
            label = {"regime": "A", "tau": tau, "machine_order": mo[0]}
            proposal: Reports = []
            for _ in range(Agt):
                levels: Report = []
                for t in range(1, T+1):
                    if tau >= 1 and t > tau:
                        # t>tau は OUT より下（提出しない扱い）→スキップ
                        continue
                    if M == 1:
                        levels.append([(0, t)])
                    else:
                        if mo[0] == "A>B":
                            levels.append([(0, t), (1, t)])
                        elif mo[0] == "B>A":
                            levels.append([(1, t), (0, t)])
                        else:  # A=B
                            levels.append([(0, t), (1, t)])
                levels.append(['OUT'])
                proposal.append(levels)
            yield (label, proposal)



def _all_rankings(items: List[Item]) -> Iterable[List[Item]]:
    # 小規模用：全順列
    return itertools.permutations(items)

def enumerate_devs_B_unconstrained(truth: Reports, M: int, T: int, K: int = 200, seed: int = 0) -> Iterable[Reports]:
    """
    B: 無制約の dev を生成。
    - T=2 のとき：全順列（(M*T)!）＋ OUT の位置を全探索（中位に挿入可能）。
    - T=3 のとき：局所スワップ＋ヒューリスティックで K 本サンプル。
    """
    rng = random.Random(seed)
    Agt = len(truth)
    items = [(m,t) for t in range(1,T+1) for m in range(M)]
    if T == 2 and M == 2:
        # 全列挙：4! = 24 と OUT の挿入位置（0..4）
        for perm in _all_rankings(items):
            for pos in range(0, len(items)+1):
                rep: Reports = []
                for _ in range(Agt):
                    levels: Report = []
                    # ここでは単純化のため、完全順序→各レベル1項目とする
                    for idx, it in enumerate(perm):
                        if idx == pos:
                            levels.append(['OUT'])
                        levels.append([it])
                    if pos == len(items):
                        levels.append(['OUT'])
                    rep.append(levels)
                yield rep
    else:
        # 近傍（隣接スワップ）＋ OUT ランダム挿入で K 本
        base = list(items)
        for _ in range(K):
            rep: Reports = []
            # ランダムに隣接スワップを複数回
            cand = base[:]
            for __ in range(rng.randint(1, 5)):
                i = rng.randrange(0, len(cand)-1)
                cand[i], cand[i+1] = cand[i+1], cand[i]
            pos = rng.randrange(0, len(cand)+1)
            for __ in range(Agt):
                levels: Report = []
                for idx, it in enumerate(cand):
                    if idx == pos:
                        levels.append(['OUT'])
                    levels.append([it])
                if pos == len(cand):
                    levels.append(['OUT'])
                rep.append(levels)
            yield rep

# 追加：単一プレイヤーの「真の選好（時間単調・一貫性あり）」をラベル付きで全列挙
def enumerate_truth_A_time_monotone(M: int, T: int):
    """
    yield: (label_dict, report_for_one_player)
      label_dict = {"tau": τ, "machine_order": "A>B"|"A=B"|"B>A"}
    ここで τ は OUT のしきい値（τ より遅い時間は OUT より下＝受け入れない）。
    ※ T=2 では τ=0 と τ=2 は等価だが、プログラム上は列挙する。
    """
    # 機械順の候補
    mo_list = ["A>B"] if M == 1 else ["A>B", "A=B", "B>A"]
    for tau in range(0, T+1):
        for mo in mo_list:
            # 単一プレイヤーの弱順序を構成
            levels: Report = []
            for t in range(1, T+1):
                if tau >= 1 and t > tau:
                    continue  # τより遅い時間は OUT の下＝提出なし
                if M == 1:
                    levels.append([(0, t)])
                else:
                    if mo == "A>B":
                        levels.append([(0, t), (1, t)])
                    elif mo == "B>A":
                        levels.append([(1, t), (0, t)])
                    else:  # A=B
                        levels.append([(0, t), (1, t)])
            levels.append(['OUT'])
            label = {"tau": tau, "machine_order": mo}
            yield (label, levels)

# 追加：2人プレイヤーの「真の選好ペア」を全列挙（直積）
def enumerate_truth_profiles_pair(M: int, T: int):
    """
    2人分の真の選好を直積で列挙。
    yield: (label_pair, reports_pair)
      label_pair = {
        "p1": {"tau": τ1, "machine_order": ...},
        "p2": {"tau": τ2, "machine_order": ...},
      }
      reports_pair = [report_p1, report_p2]
    """
    # 単体を列挙してキャッシュ
    single = list(enumerate_truth_A_time_monotone(M, T))
    for (lab1, rep1) in single:
        for (lab2, rep2) in single:
            yield (
                {"p1": lab1, "p2": lab2},
                [rep1, rep2],
            )
