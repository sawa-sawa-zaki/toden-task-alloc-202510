# -*- coding: utf-8 -*-
"""
時間単調（早いほど嬉しい）＋ mode=A（一貫性ドメイン）の
- 真の選好の全列挙（2人ペア）
- 片側だけの「mode=A 内での全嘘（12通り）」の列挙

前提：
- 機械は M=2 を想定（0: A, 1: B）。必要なら拡張してください。
- mode=A のドメインは「機械順（A>B, A=B, B>A） × tau∈{0,1,..,T}」の直積で 1人 3×(T+1) 通り。
  *T=3 なら 3×4=12 通り/人 → 2人直積で 144 プロファイル。
- tau は“時間の滑らかさ”のタグ（ここではラベル用途）。ランキング自体は「時間優先」で生成します。
  ※ 現時点では tau による順序の違いは付けません（必要になったら拡張可能）。

出力の report 形式：
- 「レベル（無差別集合）のリスト」。各レベルは [(m, t1), ...] の配列。
- 最後に 'OUT' を含めても構いません（linearize_levels 側は無視します）。
"""

from typing import Iterable, Iterator, List, Tuple, Dict, Any

MachineOrder = str  # "A>B" / "A=B" / "B>A"
Report = List[List[Tuple[int, int]]]  # レベルのリスト。各レベルは [(m, t1), ...]
PairLabel = Dict[str, Dict[str, Any]]


def _machine_orders(M: int) -> List[MachineOrder]:
    if M != 2:
        # 最小限の対応。3台以上はここを一般化してください。
        raise ValueError("This helper currently assumes M=2 (machines A/B).")
    return ["A>B", "A=B", "B>A"]


def _order_to_sequence(order: MachineOrder) -> List[List[int]]:
    """機械順を『レベル（同価）』の並びに変換。"""
    if order == "A>B":
        return [[0], [1]]
    elif order == "A=B":
        return [[0, 1]]
    elif order == "B>A":
        return [[1], [0]]
    else:
        raise ValueError(f"unknown machine order: {order}")


def make_modeA_report(order: MachineOrder, tau: int, M: int, T: int) -> Report:
    """
    mode=A の1人分の申告（report）を作る。
    ここでは「時間が早いほど嬉しい」を厳守し、各 t1=1..T について
    機械順 order に従ってレベルを積む。tau はラベルのみ（ランキングには影響させない）。
    """
    # 時間（t1）最優先、同一 t1 内で machine order
    rep: Report = []
    level_for_order = _order_to_sequence(order)  # 例: [[0],[1]] or [[0,1]] or [[1],[0]]
    for t1 in range(1, T + 1):
        for same_level in level_for_order:
            # 同一レベルなら同価（A=B のとき [(0,t1),(1,t1)]）
            rep.append([(m, t1) for m in same_level])
    # OUT を最後に付けたい場合は有りだが、linearize_levels は OUT を無視するので省略可
    return rep


def _all_modeA_one_agent(M: int, T: int) -> List[Dict[str, Any]]:
    """1人分の mode=A の全選好（ラベル＋report）を列挙。"""
    outs = []
    for order in _machine_orders(M):
        for tau in range(0, T + 1):
            rep = make_modeA_report(order, tau, M, T)
            outs.append({
                "label": {"machine_order": order, "tau": tau},
                "report": rep
            })
    return outs


def enumerate_truth_profiles_pair(M: int, T: int) -> Iterator[Tuple[PairLabel, List[Report]]]:
    """
    2人分の真の選好（mode=A）を直積で列挙。
    戻り値:
      (label_pair, [report_p1, report_p2])
      label_pair = {"p1":{"machine_order":..., "tau":...}, "p2":{...}}
    """
    one = _all_modeA_one_agent(M, T)
    for p1 in one:
        for p2 in one:
            label = {
                "p1": {"machine_order": p1["label"]["machine_order"], "tau": p1["label"]["tau"]},
                "p2": {"machine_order": p2["label"]["machine_order"], "tau": p2["label"]["tau"]},
            }
            yield (label, [p1["report"], p2["report"]])


def enumerate_devs_modeA_for_pair(
    reports_truth: List[Report], who: int, M: int, T: int
) -> Iterator[Tuple[Dict[str, Any], List[Report]]]:
    """
    与えられた真の選好（2人分 reports_truth）に対し、
    片側だけ（who=0 or 1）が mode=A の全通り（3*(T+1)）で嘘をつく組を列挙。
    戻り値:
       (label_dev, reports_dev)
       label_dev には {"regime":"A","who":who,"machine_order":..., "tau":...} を付与。
       reports_dev は [report_p1', report_p2']（片側のみ差し替え）
    """
    assert who in (0, 1)
    devs_one = _all_modeA_one_agent(M, T)
    other = 1 - who
    for cand in devs_one:
        rep_dev = [None, None]  # type: ignore
        rep_dev[who] = cand["report"]
        rep_dev[other] = reports_truth[other]
        label = {
            "regime": "A",
            "who": who,
            "machine_order": cand["label"]["machine_order"],
            "tau": cand["label"]["tau"],
        }
        yield (label, rep_dev)  # type: ignore
