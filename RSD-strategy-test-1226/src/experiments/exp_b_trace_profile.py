
"""
実験B：特定プロファイルでの詳細トレース（Phase Aのみ）

ここではまず「割当結果」を保存する最小版。
次のステップで、各期の候補集合・選択理由（val>0条件、cap、window等）をログ化していく。
"""

from __future__ import annotations
import os, json
import numpy as np
from dataclasses import asdict

from src.config.base_config import Config, ICConfig
from src.core.generator import generate_consistent_with_truncation, ranks_to_vals
from src.core.initial_rsd import run_phaseA_given_orders
from src.utils.results import make_timestamped_dir

def run():
    cfg = Config()
    icfg = ICConfig()

    out_dir = make_timestamped_dir(icfg.OUTPUT_BASE_DIR, prefix="expB_trace")
    os.makedirs(out_dir, exist_ok=True)

    # 例：真のタイプを2つ選ぶ（必要に応じて固定）
    restricted = generate_consistent_with_truncation(cfg.M, cfg.T)
    trueA = restricted[0]["ranks"]
    trueB = restricted[1]["ranks"]

    valsA = np.array(ranks_to_vals(trueA, cfg.M, cfg.T), dtype=float)
    valsB = np.array(ranks_to_vals(trueB, cfg.M, cfg.T), dtype=float)

    report_vals = np.zeros((cfg.A, cfg.M, cfg.T), dtype=float)
    report_vals[0] = valsA
    report_vals[1] = valsB

    # order sequence を固定してトレース（例：毎期 [0,1]）
    order_seq = [[0,1] for _ in range(cfg.T)]
    alloc = run_phaseA_given_orders(cfg, report_vals, order_seq)

    with open(os.path.join(out_dir, "trace.json"), "w", encoding="utf-8") as f:
        json.dump({
            "Config": asdict(cfg),
            "ICConfig": asdict(icfg),
            "true_ranks_A": list(trueA),
            "true_ranks_B": list(trueB),
            "order_seq": order_seq,
            "alloc": alloc.tolist(),
        }, f, ensure_ascii=False, indent=2)

    print(f"[expB_trace] saved to: {out_dir}")
