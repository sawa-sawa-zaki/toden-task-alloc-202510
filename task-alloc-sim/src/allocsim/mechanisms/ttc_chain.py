import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Chain:
    path: List[Tuple[int, int, int]]  # [(a, m, t), ...]
    improves: int                     # 厳密改善人数
    ends: Tuple[str, Optional[int], Optional[int]]  # ("EMPTY_K", m, t=k) or ("OUTSIDE", None, None)


def _strict_better(v_new: float, v_old: float) -> bool:
    return v_new > v_old


def _weak_better(v_new: float, v_old: float) -> bool:
    return v_new >= v_old


def _value(values: np.ndarray, a: int, m: int, t: int) -> float:
    return float(values[a, m, t])


def _neighbors_for_agent(values: np.ndarray, a: int, cur: Optional[Tuple[int, int, int]],
                         T: List[int], M: int) -> List[Tuple[int, int, int, bool]]:
    """
    エージェント a が現在 cur=(a, m0, t0)（or None=outside）から
    弱改善(=含む)で移りたい候補 (m,t) の列挙。
    戻り値は (a, m, t, is_strict)。
    """
    A, MM, TT = values.shape
    base_v = -1e18 if cur is None else _value(values, a, cur[1], cur[2])
    res = []
    for t in T:
        for m in range(M):
            v = _value(values, a, m, t)
            if v == -1:
                continue
            if _weak_better(v, base_v):
                res.append((a, m, t, _strict_better(v, base_v)))
    return res


def _machine_room_left(cap_m: np.ndarray, alloc_t: np.ndarray) -> np.ndarray:
    return cap_m - alloc_t.sum(axis=0)


def _time_total(alloc_t: np.ndarray) -> int:
    return int(alloc_t.sum())


def _find_chain_to_empty_k(values: np.ndarray, alloc_all: np.ndarray, k: int, cap_m: np.ndarray,
                           target_k: int, rng: np.random.Generator,
                           prefer_improve_over_new: bool) -> Optional[Chain]:
    """
    t>=k の全配分を使って、t=k の「空席」へ流し込む1ユニットのチェーンを探す。
    - outside（未配分）からの直接追加（禁止でない最良）をまず探す
    - その後、既配分者の厳密改善を優先して追加
    """
    T_all = list(range(k, alloc_all.shape[0]))
    A = alloc_all.shape[1]
    M = alloc_all.shape[2]

    alloc_k = alloc_all[k]
    room_time = target_k - _time_total(alloc_k)
    if room_time <= 0:
        return None
    room_m = _machine_room_left(cap_m, alloc_k)

    # outside → (a,m,k)
    if not prefer_improve_over_new:
        cand_direct = []
        for a in range(A):
            for m in range(M):
                if room_m[m] <= 0:
                    continue
                v = _value(values, a, m, k)
                if v == -1:
                    continue
                cand_direct.append((v + 1e-9 * rng.random(), a, m))
        if cand_direct:
            cand_direct.sort(reverse=True)
            v, a, m = cand_direct[0]
            return Chain(path=[(a, m, k)], improves=0, ends=("EMPTY_K", m, k))

    # 厳密改善を含むチェーン（仮実装: 一段階）
    cand_improve = []
    for a in range(A):
        for t in T_all:
            for m in range(M):
                if alloc_all[t, a, m] <= 0:
                    continue
                v_old = _value(values, a, m, t)
                for m2 in range(M):
                    v_new = _value(values, a, m2, k)
                    if v_new == -1:
                        continue
                    if _strict_better(v_new, v_old) and room_m[m2] > 0:
                        cand_improve.append((v_new - v_old, a, m, m2, t))
    if cand_improve:
        cand_improve.sort(reverse=True)
        _, a, m_old, m_new, t_old = cand_improve[0]
        return Chain(path=[(a, m_old, t_old), (a, m_new, k)], improves=1, ends=("EMPTY_K", m_new, k))

    return None


def ttc_global_with_joint_caps(values_all: np.ndarray, alloc_all: np.ndarray, k: int,
                               cap_m: np.ndarray, forecast_k: int, delta_k: int,
                               buffer_k: int, rng: np.random.Generator,
                               agent_headroom: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    t=k の供給変動（＋バッファ）を、t>=k の未実行領域でTTC風チェーンにより処理。
    """
    T, A, M = alloc_all.shape
    target_k = max(0, min(forecast_k + delta_k + buffer_k, int(cap_m.sum())))
    cur_k = int(alloc_all[k].sum())

    logs: Dict[str, Any] = {
        "target_k": int(target_k),
        "cur_k": int(cur_k),
        "chains": [],
        "k": int(k)
    }

    if cur_k == target_k:
        return alloc_all, logs

    # ===== 供給増加 =====
    if cur_k < target_k:
        need = target_k - cur_k
        rem_m_k = cap_m - alloc_all[k].sum(axis=0)

        while need > 0:
            pre_total_k = int(alloc_all[k].sum())
            pre_per_machine = (alloc_all[k].sum(axis=0)).astype(int).tolist()

            chain = _find_chain_to_empty_k(values_all, alloc_all, k, cap_m, target_k, rng, prefer_improve_over_new=True)
            if chain is None:
                break

            a, m, t = chain.path[-1]
            if agent_headroom[a] <= 0:
                break

            alloc_all[k, a, m] += 1
            agent_headroom[a] -= 1
            need -= 1

            post_total_k = int(alloc_all[k].sum())
            post_per_machine = (alloc_all[k].sum(axis=0)).astype(int).tolist()

            logs["chains"].append({
                "chain_id": len(logs["chains"]),
                "type": "increase",
                "t": int(k),
                "end": {"kind": chain.ends[0], "m": int(m), "t": int(k)},
                "path": [{"agent": int(x[0]), "m": int(x[1]), "t": int(x[2]),
                          "v": float(values_all[x[0], x[1], x[2]])} for x in chain.path],
                "participants": sorted(list({int(x[0]) for x in chain.path})),
                "strict_improves": int(chain.improves),
                "weak_only": bool(chain.improves == 0),
                "selection": {"criterion": "improvers>length>random", "tie_breaker": "random"},
                "capacity_before": {"total_k": pre_total_k, "per_machine": pre_per_machine},
                "capacity_after": {"total_k": post_total_k, "per_machine": post_per_machine},
            })

        return alloc_all, logs

    # ===== 供給減少 =====
    if cur_k > target_k:
        drop = cur_k - target_k
        items = []
        for a in range(A):
            for m in range(M):
                cnt = int(alloc_all[k, a, m])
                if cnt <= 0:
                    continue
                v = float(values_all[a, m, k])
                items.extend([(v + 1e-9 * rng.random(), a, m)] * cnt)
        items.sort()
        i = 0

        while drop > 0 and i < len(items):
            pre_total_k = int(alloc_all[k].sum())
            pre_per_machine = (alloc_all[k].sum(axis=0)).astype(int).tolist()

            _, a, m = items[i]
            alloc_all[k, a, m] -= 1
            drop -= 1

            post_total_k = int(alloc_all[k].sum())
            post_per_machine = (alloc_all[k].sum(axis=0)).astype(int).tolist()

            logs["chains"].append({
                "chain_id": len(logs["chains"]),
                "type": "decrease",
                "t": int(k),
                "end": {"kind": "OUTSIDE", "m": None, "t": None},
                "path": [{"agent": int(a), "m": int(m), "t": int(k),
                          "v": float(values_all[a, m, k])}],
                "participants": [int(a)],
                "strict_improves": 0,
                "weak_only": False,
                "selection": {"criterion": "lowest-value-first", "tie_breaker": "random"},
                "capacity_before": {"total_k": pre_total_k, "per_machine": pre_per_machine},
                "capacity_after": {"total_k": post_total_k, "per_machine": post_per_machine},
            })
            i += 1

        return alloc_all, logs
