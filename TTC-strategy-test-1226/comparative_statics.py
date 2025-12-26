import os
import sys
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import random

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config, ICConfig, generate_normal_shock
from src.generator import (
    generate_consistent_with_truncation,
    generate_all_permutations,
    generate_all_weak_orders,
    ranks_to_matrix,
    classify_lie_type
)
from src.mechanism import run_online_mechanism


# =====================================================
# 追加設定
# =====================================================

OUTSIDE_PENALTY = 10000


# =====================================================
# Utility（outside penalty 込み）
# =====================================================

def utility_with_outside_penalty(avg_alloc, true_val_matrix_a):
    outside_mask = (true_val_matrix_a == 0)
    u = float(np.sum(avg_alloc[0] * true_val_matrix_a))
    u -= float(OUTSIDE_PENALTY * np.sum(avg_alloc[0] * outside_mask))
    return u


# =====================================================
# Outcome cache 作成
# =====================================================

def build_outcome_cache(cfg, truth_matrices, strategy_matrices, ic_cfg):
    """
    (A_report_type, B_truth_type) ごとの期待配分をすべて計算
    """

    # --- A側の unique matrix 化 ---
    unique_matrices_A = []
    mat_to_unique_idx = {}

    def register(mat):
        b = mat.tobytes()
        if b not in mat_to_unique_idx:
            mat_to_unique_idx[b] = len(unique_matrices_A)
            unique_matrices_A.append(mat)
        return mat_to_unique_idx[b]

    map_truth_to_unique = {}
    map_strategy_to_unique = {}

    for i, mat in enumerate(truth_matrices):
        map_truth_to_unique[i] = register(mat)
    for i, mat in enumerate(strategy_matrices):
        map_strategy_to_unique[i] = register(mat)

    n_unique_A = len(unique_matrices_A)
    n_truth = len(truth_matrices)

    print(f"Optimized Input A: {n_unique_A} unique types")
    print(f"Total Pairs to Simulate: {n_unique_A * n_truth}")

    outcome_cache = {}
    seeds = [ic_cfg.SEED + s for s in range(ic_cfg.SAMPLES)]

    for u_idx_a, idx_b in tqdm(
        itertools.product(range(n_unique_A), range(n_truth)),
        total=n_unique_A * n_truth,
        desc="Simulating Outcomes"
    ):
        mat_a = unique_matrices_A[u_idx_a]
        mat_b = truth_matrices[idx_b]

        report_vals = np.stack([mat_a, mat_b])
        avg_alloc = np.zeros((cfg.A, cfg.M, cfg.T), dtype=float)

        for seed in seeds:
            rng = random.Random(seed)
            alloc = run_online_mechanism(cfg, report_vals, rng)
            avg_alloc += alloc

        avg_alloc /= float(ic_cfg.SAMPLES)
        outcome_cache[(u_idx_a, idx_b)] = avg_alloc

    return outcome_cache, map_truth_to_unique, map_strategy_to_unique


# =====================================================
# Comparative Statics 本体
# =====================================================

def run_experiment_scenario(
    base_cfg,
    ic_cfg,
    true_types,
    strategy_types,
    true_matrices,
    strategy_matrices,
    variable_name,
    range_values
):
    results = []

    for val in range_values:
        print(f"\n=== {variable_name} = {val} ===")

        # --- Config 更新 ---
        supply = base_cfg.BASE_SUPPLY[0]
        buffer = base_cfg.BUFFER[0]
        shock_mag = 0

        if variable_name == "Supply":
            supply = val
        elif variable_name == "Buffer":
            buffer = val
        elif variable_name == "Shock":
            shock_mag = val

        if shock_mag == 0:
            shock_prob = {0: 1.0}
        else:
            shock_prob = generate_normal_shock(shock_mag)

        cfg = Config(
            A=base_cfg.A,
            M=base_cfg.M,
            T=base_cfg.T,
            WINDOW=base_cfg.WINDOW,
            MACHINE_CAPACITY=base_cfg.MACHINE_CAPACITY,
            STRATEGY_DOMAIN=base_cfg.STRATEGY_DOMAIN,
            Q=base_cfg.Q,
            BASE_SUPPLY=[supply] * base_cfg.T,
            BUFFER=[buffer] * base_cfg.T,
            SHOCK_PROB=shock_prob
        )

        # --- Outcome cache ---
        outcome_cache, map_truth_to_unique, map_strategy_to_unique = (
            build_outcome_cache(cfg, true_matrices, strategy_matrices, ic_cfg)
        )

        # --- IC 検証 ---
        count_total = 0
        count_type_a = 0
        count_type_b = 0

        truth_indices = list(range(len(true_types)))
        target_pairs = list(itertools.product(truth_indices, repeat=2))

        for idx_true_a, idx_true_b in tqdm(
            target_pairs,
            desc="Verifying IC",
            leave=False
        ):
            val_matrix_a = true_matrices[idx_true_a]

            # honest
            u_key_truth = map_truth_to_unique[idx_true_a]
            alloc_truth = outcome_cache[(u_key_truth, idx_true_b)]
            u_truth = utility_with_outside_penalty(alloc_truth, val_matrix_a)

            # best lie
            max_u_lie = -1e100
            best_lie_idx = -1

            for idx_lie in range(len(strategy_matrices)):
                u_key_lie = map_strategy_to_unique[idx_lie]
                alloc_lie = outcome_cache[(u_key_lie, idx_true_b)]
                u_lie = utility_with_outside_penalty(alloc_lie, val_matrix_a)

                if u_lie > max_u_lie:
                    max_u_lie = u_lie
                    best_lie_idx = idx_lie

            if max_u_lie - u_truth > 1e-7:
                count_total += 1
                v_type = classify_lie_type(
                    true_types[idx_true_a]["ranks"],
                    strategy_types[best_lie_idx]["ranks"]
                )
                if v_type == "Type A":
                    count_type_a += 1
                elif v_type == "Type B":
                    count_type_b += 1

        results.append({
            "parameter": val,
            "violation_total": count_total,
            "violation_type_a": count_type_a,
            "violation_type_b": count_type_b
        })

        print(
            f"Violations: {count_total} "
            f"(Type A: {count_type_a}, Type B: {count_type_b})"
        )

    # ★★★ ここが重要 ★★★
    return pd.DataFrame(results)


# =====================================================
# main
# =====================================================

def main():
    base_cfg = Config()
    ic_cfg = ICConfig()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results_comparative", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output: {output_dir}")
    print(f"Outside penalty: -{OUTSIDE_PENALTY}")

    # --- 選好生成 ---
    true_types = generate_consistent_with_truncation(base_cfg.M, base_cfg.T)
    true_matrices = [ranks_to_matrix(pt["ranks"]) for pt in true_types]

    if base_cfg.STRATEGY_DOMAIN == "UNRESTRICTED":
        strategy_types = generate_all_permutations(base_cfg.M, base_cfg.T)
    elif base_cfg.STRATEGY_DOMAIN == "UNRESTRICTED_WEAK":
        strategy_types = generate_all_weak_orders(base_cfg.M, base_cfg.T)
    else:
        strategy_types = true_types

    strategy_matrices = [ranks_to_matrix(pt["ranks"]) for pt in strategy_types]

    # --- Experiments ---
    experiments = {
        "Supply": [1, 5, 10, 15, 20],
        "Buffer": [0, 2, 4, 6, 8, 10],
        "Shock": [0, 2, 4, 6, 8, 10]
    }

    for name, vals in experiments.items():
        df = run_experiment_scenario(
            base_cfg,
            ic_cfg,
            true_types,
            strategy_types,
            true_matrices,
            strategy_matrices,
            name,
            vals
        )
        df.to_csv(os.path.join(output_dir, f"exp_{name.lower()}.csv"), index=False)

    print("Done.")


if __name__ == "__main__":
    main()
