# --- use canonical mechanism implementations ---
import numpy as np
from typing import Any, Dict, Tuple, List, Callable, Optional

# ✅ 実在する関数を直指定
from src.allocsim.mechanisms.rsd_global import rsd_global_with_joint_caps
from src.allocsim.mechanisms.ttc_chain import ttc_global_with_joint_caps

import importlib

# 差分：ttc_global_with_joint_caps を呼ぶ（kのときだけ）
from .mechanisms.ttc_chain import ttc_global_with_joint_caps

def run_pipeline_for_one_seed(cfg, values_all, market, rng):
    A, M, T = values_all.shape
    units_demand = np.full(A, market.units_demand, dtype=int)

    # RSD（予測-バッファ、二層制約）
    cap_t_rsd = np.array(market.cap_vec_for_rsd(), dtype=int)
    cap_m = np.array(market.machine_capacity, dtype=int)
    from .mechanisms.rsd_global import rsd_global_with_joint_caps
    alloc_all, order, rsd_choice_log = rsd_global_with_joint_caps(
        values_all, cap_t_rsd.copy(), cap_m.copy(), units_demand.copy(), rng
    )
    rsd_snapshot_allT = alloc_all.copy()

    # headroom
    total_assigned_per_agent = alloc_all.sum(axis=(0, 2))
    headroom = units_demand - total_assigned_per_agent

    trace = []

    for k in range(T):
        # t<k は固定、t>=k は未実行領域
        # k の実質変動は delta[k]+buffer[k] を使う（あなたの仕様）
        delta_k = 0
        for s in market.supply_shock_by_t:
            if s["t"] == k:
                delta_k += int(s["delta"])
        buffer_k = market.buffer_t(k)
        forecast_k = market.forecast_t(k)

        alloc_before = alloc_all.copy()

        alloc_all, tlogs = ttc_global_with_joint_caps(
            values_all, alloc_all, k, cap_m.copy(), forecast_k, delta_k, buffer_k,
            rng, headroom.copy()
        )

        # headroom更新（t>=k 合計が変わる）
        total_assigned_per_agent_new = alloc_all.sum(axis=(0, 2))
        headroom = units_demand - total_assigned_per_agent_new

        trace.append({
            "t": int(k),
            "rsd_alloc_allT": rsd_snapshot_allT,   # 固定スナップショット
            "after_alloc_allT": alloc_all.copy(),
            "pick_order": [int(x) for x in order],
            "rsd_choice_log": rsd_choice_log,
            "ttc_logs": tlogs,
            "forecast_total_t": int(forecast_k),
            "buffer_t": int(buffer_k),
            "target_total_after": int(tlogs.get("target_k", 0)),
        })

    return {"allocation": alloc_all, "remaining_demand": headroom, "trace": trace}

def rsd_initial_global(values_all: np.ndarray, market, rng) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    シンプルなRSD実装（需要=units_demand、-1は不可、RSDは予測-バッファ分だけ使う）
    - values_all: (A,M,T)  申告値（-1 可）
    - market: Market（machine_capacity, supply_forecast_by_t, buffer_by_t, units_demand を参照）
    - rng: np.random.Generator
    戻り:
      allocation: (T,A,M) 0/1（需要>1なら複数回入る可能性あり）
      logs: {"order": [...], "notes": "..."}
    """
    A = values_all.shape[0]
    M = market.num_machines
    T = market.time_horizon
    demand_each = market.units_demand

    # 各時刻の合計供給上限（予測-バッファ、かつ機械の合計キャパを超えない）
    time_cap = []
    sum_machine_cap = int(np.sum(market.machine_capacity))
    for t in range(T):
        # RSD段階ではバッファは使わない（バッファはTTC用）
        cap_t = int(max(0, market.supply_forecast_by_t[t] - market.buffer_by_t[t]))
        time_cap.append(min(cap_t, sum_machine_cap))
    time_cap = np.array(time_cap, dtype=int)

    # 各 (t,m) の残キャパ（機械キャパ＝時刻ごとに同じ）
    rem_tm = np.zeros((T, M), dtype=int)
    for m in range(M):
        rem_tm[:, m] = market.machine_capacity[m]

    # 出力配列
    alloc = np.zeros((T, A, M), dtype=int)

    # RSD順序
    order = rng.permutation(A).tolist()

    # -1のスロットは選好から除外
    mask_valid = (values_all >= 0.0)

    # 各エージェントを順に、需要を満たすまで貪欲に割当
    for a in order:
        need = int(demand_each)
        if need <= 0:
            continue

        # 候補スロットを (value DESC, t ASC, m ASC) で並べる
        cands = []
        for t in range(T):
            for m in range(M):
                if not mask_valid[a, m, t]:
                    continue
                v = float(values_all[a, m, t])
                cands.append(( -v, t, m ))  # value降順にしたいので符号反転
        cands.sort()

        for negv, t, m in cands:
            if need <= 0:
                break
            # 残キャパと時刻合計制約の両方を見る
            if rem_tm[t, m] <= 0:
                continue
            if time_cap[t] <= 0:
                continue
            # 割当
            alloc[t, a, m] += 1
            rem_tm[t, m] -= 1
            time_cap[t] -= 1
            need -= 1

    logs = {"order": order, "notes": "RSD on forecast minus buffer; -1 slots ignored"}
    return alloc, logs

def _resolve_func(module_path: str, candidates: list[str]) -> Callable[..., Any]:
    """
    module_path を import し、candidates に挙がっている関数名のうち
    最初に見つかったものを返す。見つからない場合は、そのモジュール内の
    関数一覧を表示して分かりやすく落とす。
    """
    mod = importlib.import_module(module_path)
    for name in candidates:
        if hasattr(mod, name) and callable(getattr(mod, name)):
            return getattr(mod, name)

    # ヒントを出す（そのモジュールに実際にある関数名を列挙）
    funcs = [n for (n, obj) in inspect.getmembers(mod, inspect.isfunction) if obj.__module__ == mod.__name__]
    raise AttributeError(
        f"[resolve error] None of {candidates} found in {module_path}.\n"
        f"Available functions in {module_path}: {funcs}"
    )


# --- RSD from mechanisms (rsd_global.py / rsd_multi.py) ---
def _get_rsd_func() -> Callable[..., Tuple[np.ndarray, Dict[str, Any]]]:
    # よくある呼び名候補を順に試す
    search_spaces = [
        ("src.allocsim.mechanisms.rsd_global", ["rsd_initial_global", "rsd_global", "run_rsd_global"]),
        ("src.allocsim.mechanisms.rsd_multi",  ["rsd_initial_multi", "rsd_multi", "run_rsd_multi"]),
    ]
    last_err = None
    for module_path, candidates in search_spaces:
        try:
            func = _resolve_func(module_path, candidates)
            # print(f"[use] RSD: {module_path}.{func.__name__}")
            return func
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("RSD function not found")

# --- TTC from mechanisms (ttc_chain.py) ---
def _get_ttc_func() -> Callable[..., Tuple[np.ndarray, Any]]:
    search_spaces = [
        ("src.allocsim.mechanisms.ttc_chain",
         ["ttc_adjust_by_timeslot_total", "ttc_global", "run_ttc", "ttc_chain_global"]),
    ]
    last_err = None
    for module_path, candidates in search_spaces:
        try:
            func = _resolve_func(module_path, candidates)
            # print(f"[use] TTC: {module_path}.{func.__name__}")
            return func
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("TTC function not found")

def run_rsd_only(values_all: np.ndarray, market, rng, active_times=None):
    """
    RSD＝rsd_global_with_joint_caps を使用。
    active_times 以外の時刻は cap_t=0 にして RSD 対象外にする。
    """
    A, M, T = values_all.shape

    sum_mcap = int(np.sum(market.machine_capacity))
    cap_t = np.array([
        max(0, int(market.supply_forecast_by_t[t]) - int(market.buffer_by_t[t]))
        for t in range(T)
    ], dtype=int)
    cap_t = np.minimum(cap_t, sum_mcap)

    if active_times is not None:
        mask = np.zeros(T, dtype=bool)
        mask[active_times] = True
        for t in range(T):
            if not mask[t]:
                cap_t[t] = 0

    cap_m = np.array(market.machine_capacity, dtype=int)
    units_demand = np.full(A, int(market.units_demand), dtype=int)

    alloc, order, rsd_logs = rsd_global_with_joint_caps(
        values_all=values_all,
        cap_t=cap_t,
        cap_m=cap_m,
        units_demand=units_demand,
        rng=rng,
    )
    logs = {"order": order, "raw": rsd_logs}
    return {"allocation": alloc, "logs": {"rsd": logs}}
def run_ttc_only(values_all: np.ndarray, market, rng, init_alloc: np.ndarray):
    """
    TTC＝ttc_global_with_joint_caps を時刻 k ごとに適用。
    """
    T, A, M = init_alloc.shape
    alloc = init_alloc.copy()
    cap_m = np.array(market.machine_capacity, dtype=int)

    delta_by_t = np.zeros(T, dtype=int)
    for rec in getattr(market, "supply_shock_by_t", []):
        delta_by_t[int(rec["t"])] += int(rec["delta"])

    chains: List[Dict[str, Any]] = []
    for k in range(T):
        forecast_k = int(market.supply_forecast_by_t[k])
        delta_k    = int(delta_by_t[k])
        buffer_k   = int(market.buffer_by_t[k])

        assigned_per_agent = alloc.sum(axis=(0, 2))  # (A,)
        agent_headroom = np.maximum(
            0,
            np.full(A, int(market.units_demand), dtype=int) - assigned_per_agent.astype(int)
        )

        alloc, log_k = ttc_global_with_joint_caps(
            values_all=values_all,
            alloc_all=alloc,
            k=k,
            cap_m=cap_m,
            forecast_k=forecast_k,
            delta_k=delta_k,
            buffer_k=buffer_k,
            rng=rng,
            agent_headroom=agent_headroom,
        )
        chains.append({"k": k, "log": log_k})

    return {"allocation": alloc, "logs": {"ttc_chains": chains}}

