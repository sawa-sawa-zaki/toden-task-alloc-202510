# src/allocsim/strategy/weak_orders.py
from typing import List, Tuple, Union, Iterable
import itertools

Item  = Tuple[int, int]                   # (m, t1-based)
Level = List[Union[Item, str]]            # レベル集合。'OUT' を含み得る
Report = List[Level]                      # 弱順序（レベルのリスト）
Reports = List[Report]

def _ordered_partitions(seq):
    """
    与えられた要素列 seq を、順序付きブロック分割の全通りに列挙する。
    例: [a,b,c] -> [[ [a,b,c] ],
                    [ [a],[b,c] ], [ [b],[a,c] ], [ [c],[a,b] ],
                    [ [a,b],[c] ], [ [a,c],[b] ], [ [b,c],[a] ],
                    [ [a],[b],[c] ] の各ブロック順の全並べ替え …]
    """
    n = len(seq)
    # set partitions（順序なしブロック分割）を先に作り、次にブロック順を全並べ替え
    # 小規模（n=4）前提なので素直に実装
    def _set_partitions(seq):
        if not seq:
            yield []
            return
        first, rest = seq[0], seq[1:]
        for parts in _set_partitions(rest):
            # 既存ブロックに入れる
            for i in range(len(parts)):
                yield parts[:i] + [parts[i] + [first]] + parts[i+1:]
            # 新しいブロックを作る
            yield [[first]] + parts

    seen = set()
    for blocks in _set_partitions(list(seq)):
        # 各ブロック内の要素順は固定（辞書順に正規化）
        norm = tuple(tuple(sorted(b)) for b in blocks)
        if norm in seen:
            continue
        seen.add(norm)
        # ブロック順の全並べ替え
        for order in itertools.permutations(norm):
            yield [list(b) for b in order]

def enumerate_weak_orders_for_cells(M: int, T: int) -> Iterable[Report]:
    """
    M×T 個のセル (m,t1-based) に対する弱順序（ties可）を全列挙し、Report 形式で返す。
    ※ 実運用は M=2, T=2 を想定（n=4 → 75通り）。
    """
    items = [(m, t1) for t1 in range(1, T+1) for m in range(M)]
    # items をインデックス化（順序の一意性確保用）
    id_of = {items[i]: i for i in range(len(items))}

    for blocks in _ordered_partitions(list(range(len(items)))):
        # blocks: [[idx, ...], [idx, ...], ...] （上位レベル → 下位レベル の順）
        levels: Report = []
        for blk in blocks:
            level: Level = [items[idx] for idx in blk]
            levels.append(level)
        levels.append(['OUT'])  # OUT は末尾に固定
        yield levels

def enumerate_truth_profiles_weak_pair(M: int, T: int) -> Iterable[Tuple[dict, Reports]]:
    """
    2人分の真の弱順序（ties可）を直積で全列挙。
    返り値: (label_pair, reports_pair)
      label_pair = {"p1_id": i, "p2_id": j}
      reports_pair = [report_p1, report_p2]
    """
    all_reports = list(enumerate_weak_orders_for_cells(M, T))
    for i, r1 in enumerate(all_reports):
        for j, r2 in enumerate(all_reports):
            yield ({"p1_id": i, "p2_id": j}, [r1, r2])
