# src/allocsim/online_rsd_ttc.py
from typing import List, Tuple, Union, Dict, Any
import numpy as np
from dataclasses import dataclass

Item = Tuple[int, int]           # (m, t)
Level = List[Union[Item, str]]   # レベル集合。'OUT' を含む
Report = List[Level]
Reports = List[Report]

@dataclass
class Config:
    A: int               # players
    M: int               # machines
    T: int               # timeslots
    units_demand: List[int]  # q_i per player
    barS: List[int]      # predicted supply per t
    buffer: List[int]    # integer buffer per t
    window: int          # time window length (e.g., 2 or 3)
    shock_p: float       # P(Δ=+1)=P(Δ=-1)=p/2 (期待0)
    shock_seed: int = 0

def draw_supply_realization(barS_t: int, p: float, rng: np.random.Generator) -> int:
    r = rng.random()
    if r < p/2:   return max(0, barS_t + 1)
    if r < p:     return max(0, barS_t - 1)
    return barS_t

def compute_utilities(values_all: np.ndarray, alloc: np.ndarray) -> np.ndarray:
    """
    values_all: shape (A, M, T), 価値
    alloc:      shape (T, A, M), 割当(0/1)
    return:     shape (A,), 各人の効用（加法）
    """
    T, A, M = alloc.shape
    assert values_all.shape == (A, M, T)
    # (T,A,M) x (A,M,T) -> (A,)
    u = np.zeros(A, dtype=float)
    for t in range(T):
        for a in range(A):
            for m in range(M):
                if alloc[t,a,m]:
                    u[a] += values_all[a,m,t]
    return u

def _tie_break(level: Level) -> List[Union[Item,str]]:
    # レベル内の順序はそのまま（既に一貫性を保って与えられる想定）
    return list(level)

def rsd_step(cfg: Config, reports: Reports, alloc: np.ndarray, residual: np.ndarray, rsd_order: List[int], t0: int):
    """
    1ウィンドウのRSD。
    - residual[t0'] は t0' の残余総量（t0' は 0-based）。
    - reports 内の (m, t) は t が 1..T の 1-based なので、参照時に 0-based へ変換する。
    - プレイヤー i はウィンドウ内で上位から needs_i 本選べる。
    """
    T, A, M = alloc.shape[0], cfg.A, cfg.M
    w = cfg.window
    # 0-based のウィンドウ [t0, ..., min(t0+w-1, T-1)]
    window = list(range(t0, min(t0 + w, cfg.T)))

    for i in rsd_order:
        # 需要は総量 q_i。既に確保済み（全期間）の分を引いて残りをこのウィンドウで取りに行く
        need_total = cfg.units_demand[i]
        have_total = int(alloc[:, i, :].sum())
        need = max(0, need_total - have_total)
        if need <= 0:
            continue

        # 弱順序のレベルごとにスキャン
        for level in reports[i]:
            cand = _tie_break(level)
            for it in cand:
                if it == 'OUT':
                    continue
                m, t1 = it          # t1 は 1..T（1-based）
                tau0 = t1 - 1       # 0-based に変換
                if tau0 not in window:
                    continue
                if residual[tau0] <= 0:
                    continue
                # 配分
                alloc[tau0, i, m] += 1
                residual[tau0] -= 1
                need -= 1
                if need == 0:
                    break
            if need == 0:
                break

def _best_improvement_edge(values_all: np.ndarray, alloc: np.ndarray, i: int, current_items: List[Tuple[int,int]]) -> Union[Tuple[int,int], None]:
    """
    TTCで使う：プレイヤー i が「現在持っている束より厳密に好む」候補を探す簡易版。
    ここでは単純に 未取得の (m,t) で価値差が最も大きいものを選ぶ。
    """
    A, M, T = values_all.shape
    have_set = set(current_items)
    best = None
    best_gain = 0.0
    cur_val = sum(values_all[i,m,t] for (m,t) in current_items)
    for t in range(T):
        for m in range(M):
            if alloc[t,i,m]:  # 既に持っている
                continue
            gain = values_all[i,m,t]
            if gain > 0 and (gain > best_gain):
                best_gain = gain
                best = (m,t)
    return best

def ttc_promote_one(cfg: Config, values_all: np.ndarray, alloc: np.ndarray, target_t: int) -> bool:
    """
    t の席を +1 本増やす繰り上げチェーンを1本だけ処理（簡易TTC）。
    - 空いた1席（ダミー席）が誰かに渡る→その人がより好む席へ→…の連鎖を1本確定。
    ※ 小規模前提の簡易実装（A=2, M<=2, T<=3 くらいを想定）
    """
    A, M, T = values_all.shape
    # ダミー空席: target_t に1本
    # もっとも得をするプレイヤーを選んで与える
    best_i, best_m, best_gain = None, None, -1e9
    for i in range(A):
        for m in range(M):
            g = values_all[i,m,target_t]
            if g > best_gain and alloc[target_t,i,m] == 0:
                best_gain = g; best_i = i; best_m = m
    if best_i is None:
        return False
    # 与える
    alloc[target_t, best_i, best_m] += 1
    return True

def ttc_evict_one(cfg: Config, values_all: np.ndarray, alloc: np.ndarray, target_t: int) -> bool:
    """
    t の席を -1 本減らす追い出しチェーンを1本だけ処理（簡易TTC）。
    - もっとも“優先が低い”席（ここでは価値が最も低そうな席）を落とし、
      そのプレイヤーが次善を取りに行く（ここでは簡易に OUT 落ちでもOK）。
    """
    A, M, T = values_all.shape
    # 最も価値の小さい (i,m) を 1 つ落とす
    cand = []
    for i in range(A):
        for m in range(M):
            if alloc[target_t,i,m] > 0:
                cand.append((values_all[i,m,target_t], i, m))
    if not cand:
        return False
    cand.sort(key=lambda x: x[0])   # 価値が小さい順
    _, i_drop, m_drop = cand[0]
    alloc[target_t, i_drop, m_drop] -= 1
    # 次善へ移動（簡易：ここでは何もしない＝最終的に OUT）
    return True

def run_online_rsd_ttc(
    cfg: Config,
    values_all: np.ndarray,  # shape (A,M,T)
    reports: Reports,        # 真/虚偽どちらも可（弱順序）
    seed: int = 0
) -> Dict[str, Any]:
    """
    時間順に:
      RSD(tildeS_t) -> S_t 観測 -> (B_t 先行) -> TTC(+/-差分本数)
    を走らせ、最終 alloc とログを返す。
    RSD順は各 t でランダム化する（seed に依存）。
    """
    A, M, T = cfg.A, cfg.M, cfg.T
    assert values_all.shape == (A, M, T)

    rng = np.random.default_rng(cfg.shock_seed + seed)
    alloc = np.zeros((T, A, M), dtype=int)
    logs: Dict[str, Any] = {"rsd_order": [], "s_real": [], "ttc": []}

    for t in range(1, T+1):
        idx = t-1
        barS_t = cfg.barS[idx]
        B_t    = cfg.buffer[idx]
        eff    = max(0, barS_t - B_t)      # 実効供給（RSDで使用）

        # --- RSD 順をこの t でランダム化 ---
        rsd_order = rng.permutation(A).tolist()
        logs["rsd_order"].append(rsd_order)

        # 残余（このウィンドウで使う t' の残余総量）を初期化
        residual = np.zeros(T, dtype=int)
        residual[idx] = eff

        # --- RSD（需要サイズ分だけ確保） ---
        rsd_step(cfg, reports, alloc, residual, rsd_order, idx)

        # --- 実供給を観測 ---
        S_t = draw_supply_realization(barS_t, cfg.shock_p, rng)
        logs["s_real"].append(S_t)

        # --- バッファ先行で差分本数を算出 ---
        if S_t >= eff:
            inc = S_t - eff
            use = min(B_t, inc)
            inc -= use
            # +inc 本の繰り上げチェーン
            for _ in range(inc):
                ok = ttc_promote_one(cfg, values_all, alloc, idx)
                logs["ttc"].append({"t": t, "type": "promote", "ok": bool(ok)})
        else:
            dec = eff - S_t
            absorbed = min(B_t, dec)
            dec -= absorbed
            # -dec 本の追い出しチェーン
            for _ in range(dec):
                ok = ttc_evict_one(cfg, values_all, alloc, idx)
                logs["ttc"].append({"t": t, "type": "evict", "ok": bool(ok)})

    return {"allocation": alloc, "logs": logs}


def evaluate_regret_once(
    cfg: Config,
    values_all: np.ndarray,
    reports_truth: Reports,
    reports_dev: Reports,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1回のショック実現での（真 vs dev）各プレイヤー効用を返す。
    """
    out_truth = run_online_rsd_ttc(cfg, values_all, reports_truth, seed=seed)
    out_dev   = run_online_rsd_ttc(cfg, values_all, reports_dev,   seed=seed)
    u_truth = compute_utilities(values_all, out_truth["allocation"])
    u_dev   = compute_utilities(values_all, out_dev["allocation"])
    return u_truth, u_dev

def evaluate_ic(
    cfg: Config,
    values_all: np.ndarray,
    reports_truth: Reports,
    dev_generator,              # イテレータ: (label_dict, reports_dev) を yield
    samples: int = 200,
    seed0: int = 0,
    examples_per_agent: int = 3  # ログに残す “場面” の最大数（各エージェント）
) -> Dict[str, Any]:
    """
    期待に関する ε-IC 検証 + どんな場面で嘘が効いたかの記録。
    - ショックを 'samples' 回サンプル（seed0+i）し、E[u] を推定。
    - dev 候補を順に評価し、regret = max_dev E[u_dev] - E[u_truth] を返す。
    - 追加で、少数のサンプルで “嘘が効いた場面” のログを保存（供給実現・RSD順・devラベル等）。
    """
    A = cfg.A

    # --- 1) まず真実の期待効用 E[u_truth] を推定 ---
    EU_truth = np.zeros(A, dtype=float)
    for k in range(samples):
        out_truth = run_online_rsd_ttc(cfg, values_all, reports_truth, seed=(seed0+k))
        u_truth = compute_utilities(values_all, out_truth["allocation"])
        EU_truth += u_truth
    EU_truth /= samples

    # --- 2) dev候補を評価（平均） ---
    best = EU_truth.copy()
    best_dev_label = None
    num_dev = 0
    for item in dev_generator:
        num_dev += 1
        if isinstance(item, tuple) and len(item) == 2:
            label, rep_dev = item
        else:
            # 後方互換：ラベルがない生成器にも対応
            label = {"regime": "unknown"}
            rep_dev = item

        EU_dev = np.zeros(A, dtype=float)
        for k in range(samples):
            out_dev = run_online_rsd_ttc(cfg, values_all, rep_dev, seed=(seed0+k))
            u_dev = compute_utilities(values_all, out_dev["allocation"])
            EU_dev += u_dev
        EU_dev /= samples

        better = EU_dev > best
        if better.any():
            best = np.where(better, EU_dev, best)
            best_dev_label = label

    regret = best - EU_truth

    # --- 3) “嘘が効いた場面” をログする（軽いコストで少数サンプルだけ） ---
    examples = []  # {agent, delta, label, s_real, rsd_order_by_t} を蓄積
    # 各エージェントごとに examples_per_agent 件まで
    per_agent_count = [0] * A

    if best_dev_label is not None:
        # best_dev_label に対応する dev レポートをもう一度作り直す必要があるので、
        # dev_generator を走り直して一致するものを取得
        chosen_dev = None
        for item in dev_generator:
            if isinstance(item, tuple) and len(item) == 2:
                label, rep_dev = item
            else:
                label = {"regime": "unknown"}
                rep_dev = item
            if label == best_dev_label:
                chosen_dev = rep_dev
                break

        if chosen_dev is not None:
            # いくつかの seed で、truth vs chosen_dev を比較して、
            # 改善が出たサンプルの “場面（供給・RSD順）” を保存
            for k in range(1000):  # 上限（必要十分に小さいループ）
                if all(c >= examples_per_agent for c in per_agent_count):
                    break
                seed = seed0 + 10_000 + k  # 本推定に使った領域と分離
                out_truth = run_online_rsd_ttc(cfg, values_all, reports_truth, seed=seed)
                out_dev   = run_online_rsd_ttc(cfg, values_all, chosen_dev,     seed=seed)
                u_truth = compute_utilities(values_all, out_truth["allocation"])
                u_dev   = compute_utilities(values_all, out_dev["allocation"])
                delta = u_dev - u_truth
                for i in range(A):
                    if delta[i] > 1e-12 and per_agent_count[i] < examples_per_agent:
                        examples.append({
                            "agent": int(i),
                            "delta": float(delta[i]),
                            "label": best_dev_label,
                            "s_real": out_dev["logs"]["s_real"],              # その時の実供給 S_t
                            "rsd_order_by_t": out_dev["logs"]["rsd_order"],   # 各 t の RSD順
                        })
                        per_agent_count[i] += 1
                        # 全員分集まったら抜ける
                        if all(c >= examples_per_agent for c in per_agent_count):
                            break

    return {
        "EU_truth": EU_truth.tolist(),
        "EU_best_dev": best.tolist(),
        "regret_per_agent": regret.tolist(),
        "regret_mean": float(np.mean(regret)),
        "regret_max": float(np.max(regret)),
        "num_dev_evaluated": num_dev,
        "best_dev_label": best_dev_label,
        "examples": examples,   # ← “嘘が効いた場面”の具体ログ
    }
