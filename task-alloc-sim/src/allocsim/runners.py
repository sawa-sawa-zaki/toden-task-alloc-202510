import csv
import os
import time
import string
import numpy as np

from .utils import seed_all
from .market import Market
from .prefs.generators import generate_linear_values
from .prefs.loaders import load_linear_values_json
from .metrics import compute_metrics
from .process import run_pipeline_for_one_seed


def _agent_label(i: int) -> str:
    letters = string.ascii_uppercase
    return letters[i] if i < len(letters) else f"{letters[i % 26]}{i // 26}"


def _format_3d_alloc(alloc_TAM: np.ndarray) -> str:
    T, A, M = alloc_TAM.shape
    lines = []
    for a in range(A):
        lines.append(f"  参加者{_agent_label(a)}:")
        mat = alloc_TAM[:, a, :]
        for t in range(T):
            lines.append(f"    t={t+1}: {mat[t].tolist()}")
    return "\n".join(lines)


def _write_game_log(path: str, cfg, market: Market, values_all: np.ndarray, trace: list):
    A = market.num_agents
    M = market.num_machines
    T = market.time_horizon

    forecast_supply = [int(x) for x in market.supply_forecast_by_t]
    buffer_vec = [int(x) for x in (market.buffer_by_t or [0] * T)]
    cap_used_for_rsd = [max(0, f - b) for f, b in zip(forecast_supply, buffer_vec)]
    machine_caps = list(market.machine_capacity)
    shock_rows = [f"(t={s['t']+1}, delta={s['delta']})" for s in market.supply_shock_by_t]

    with open(path, "w", encoding="utf-8") as f:
        f.write("ゲームのフォーマット：\n")
        f.write(f"- 計算機数：[{M}]\n")
        f.write(f"- 時間帯数：[{T}]\n")
        f.write(f"- 参加者数：[{A}]\n")

        for a in range(A):
            vals = np.round(values_all[a], 2)
            f.write(
                f"- 参加者{_agent_label(a)}について："
                f"[{np.array(vals).tolist()}][需要={market.units_demand}]\n"
            )

        f.write(f"- マシン別キャパシティ：{machine_caps}\n")
        f.write(f"- 電力供給予測：{forecast_supply}\n")
        f.write(f"- バッファ：{buffer_vec}\n")
        f.write(f"- RSD使用上限（予測-バッファ）：{cap_used_for_rsd}\n")
        f.write(f"- 変動リスト：{', '.join(shock_rows) if shock_rows else '[]'}\n\n")

        rsd_order = trace[0]["pick_order"]
        rsd_choice_log = trace[0].get("rsd_choice_log", [])
        f.write("RSD優先順序: " + " → ".join([f"Agent{a}" for a in rsd_order]) + "\n")
        f.write("RSD選択過程（Serial Dictatorship）:\n")
        for entry in rsd_choice_log:
            a, t, m, v = entry["agent"], entry["time"], entry["machine"], entry["value"]
            f.write(f"  Step{entry['step']:>3}: Agent{a} → (t={t+1}, m={m+1}, v={v:.2f})\n")
        f.write("\n")

        f.write("各ステップでの配分結果:\n")
        f.write("- t=1 RSD結果（全時間帯）\n")
        f.write(_format_3d_alloc(np.array(trace[0]["rsd_alloc_allT"])) + "\n")

        for rec in trace:
            t = rec["t"]
            ttc_logs = rec.get("ttc_logs", {})
            cur_k = ttc_logs.get("cur_k", None)
            target_k = ttc_logs.get("target_k", None)
            if cur_k is not None and target_k is not None:
                f.write(f"- t={t+1} TTC目標: {cur_k}→{target_k}\n")
            else:
                f.write(f"- t={t+1} 変動後\n")
            f.write(_format_3d_alloc(np.array(rec["after_alloc_allT"])) + "\n")

        # === TTCチェーン処理ログ ===
        f.write("\nTTCチェーン処理ログ:\n")
        for rec in trace:
            t = rec["t"]
            tlogs = rec.get("ttc_logs", {})
            chains = tlogs.get("chains", [])
            if not chains:
                continue
            f.write(f"  [t={t+1}] チェーン数: {len(chains)}\n")
            for ch in chains:
                kind = ch["type"]
                cid = ch["chain_id"]
                before = ch["capacity_before"]
                after = ch["capacity_after"]
                end = ch["end"]["kind"]
                improves = ch["strict_improves"]
                parts = ",".join([f"A{p}" for p in ch["participants"]])
                f.write(
                    f"    - chain#{cid} type={kind} end={end} "
                    f"improvers={improves} participants=[{parts}] "
                    f"cap {before['total_k']}→{after['total_k']} perM {before['per_machine']}→{after['per_machine']}\n"
                )
                for step_i, step in enumerate(ch["path"]):
                    f.write(
                        f"        step{step_i}: Agent{step['agent']} → (t={step['t']+1}, m={step['m']+1}, v={step['v']:.2f})\n"
                    )
        f.write("---\n")


def run_repeats(cfg, run_dir: str):
    A = cfg.market.num_agents
    M = cfg.market.num_machines
    T = cfg.market.time_horizon

    out_csv = os.path.join(run_dir, "metrics.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["rep", "welfare", "fill_ratio", "utilization", "assigned",
                        "demand", "supply", "seed"],
        )
        w.writeheader()

    for rep in range(cfg.repeats):
        seed_all(cfg.seed + rep)
        rng = np.random.default_rng(cfg.seed + rep)

        if cfg.prefs.type == "json_linear_values":
            values_all = load_linear_values_json(cfg.prefs.path, A, M, T, allow_ties=cfg.prefs.allow_ties)
        else:
            values_all = generate_linear_values(
                num_agents=A, num_machines=M, time_horizon=T,
                dist=cfg.prefs.value_dist, mean=cfg.prefs.mean, std=cfg.prefs.std,
                allow_ties=cfg.prefs.allow_ties, tie_round=cfg.prefs.tie_round,
                per_agent_correlation=cfg.prefs.per_agent_correlation,
                seed=cfg.prefs.seed + rep,
            )

        market = Market(
            num_agents=A, num_machines=M, time_horizon=T,
            units_demand=cfg.market.units_demand,
            supply_forecast_by_t=list(cfg.market.supply_forecast_by_t),
            buffer_by_t=list(cfg.market.buffer_by_t) if getattr(cfg.market, "buffer_by_t", None) else [0] * T,
            machine_capacity=list(cfg.market.machine_capacity),
            supply_shock_by_t=[dict(s) for s in cfg.market.supply_shock_by_t],
        )

        t0 = time.time()
        result = run_pipeline_for_one_seed(cfg, values_all, market, rng)
        wall_ms = (time.time() - t0) * 1000.0

        metrics = compute_metrics(values_all, result["allocation"], market)
        metrics.update({"rep": rep, "seed": cfg.seed + rep})
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            w.writerow(metrics)

        log_path = os.path.join(run_dir, f"game_log_rep{rep}.txt")
        _write_game_log(log_path, cfg, market, values_all, result["trace"])

        if (rep + 1) % max(1, cfg.log_interval) == 0:
            print(
                f"[{rep+1}/{cfg.repeats}] "
                f"welfare={metrics['welfare']:.3f}, fill={metrics['fill_ratio']:.3f}, "
                f"supply={metrics['supply']}, assigned={metrics['assigned']} "
                f"({wall_ms:.1f} ms) -> log: {os.path.basename(log_path)}"
            )
