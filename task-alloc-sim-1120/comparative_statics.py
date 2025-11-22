import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import random

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import Config, generate_normal_shock
from src.generator import (
    generate_consistent_with_truncation,
    generate_all_permutations, 
    generate_all_weak_orders, 
    ranks_to_matrix,
    classify_lie_type
)
# from src.solver import get_exact_expected_utility # ★削除: ソルバーは使わない
from src.mechanism import run_online_mechanism    # ★追加: 直接シミュレーションする

# --- 実験設定 ---
SAMPLE_SIZE = 20          
MC_SAMPLES = 30           # ★追加: モンテカルロ試行回数 (30~50で十分傾向は見える)
DEFAULT_DEMAND = [20, 20]
DEFAULT_SUPPLY = 10
DEFAULT_BUFFER = 5
DEFAULT_SHOCK_MAG = 5
STRATEGY_MODE = "RESTRICTED"

# ★新規追加: モンテカルロ法でEUを計算するヘルパー関数
def calculate_eu_monte_carlo(cfg, report_vals, true_vals_a, true_vals_b, samples, seed_base):
    """
    シミュレーションを複数回行って平均利得(EU)を出す
    """
    total_u_a = 0.0
    
    # シードを固定して再現性を担保しつつ、試行ごとにランダム性を出す
    # (Configが変わっても同じ乱数列を使うことで比較可能性を高める)
    for i in range(samples):
        rng = random.Random(seed_base + i)
        alloc = run_online_mechanism(cfg, report_vals, rng)
        
        # Agent Aの利得だけ計算すればよい
        u_a = np.sum(alloc[0] * true_vals_a)
        total_u_a += u_a
        
    return total_u_a / samples

def run_experiment_scenario(true_types, strategy_types, true_matrices, strategy_matrices, target_profiles, variable_name, range_values):
    results = []
    print(f"--- Experiment: Varying {variable_name} (Mode: {STRATEGY_MODE}, MC={MC_SAMPLES}) ---")
    
    for val in tqdm(range_values, desc=f"Simulating {variable_name}"):
        supply = DEFAULT_SUPPLY
        buffer = DEFAULT_BUFFER
        shock_mag = DEFAULT_SHOCK_MAG
        
        if variable_name == "Supply": supply = val
        elif variable_name == "Buffer": buffer = val
        elif variable_name == "Shock": shock_mag = val
            
        if shock_mag == 0:
            shock_prob = {0: 1.0}
        else:
            shock_prob = generate_normal_shock(shock_mag)

        cfg = Config(
            A=2, M=2, T=3, WINDOW=2,
            MACHINE_CAPACITY=20,
            STRATEGY_DOMAIN=STRATEGY_MODE,
            Q=DEFAULT_DEMAND,
            BASE_SUPPLY=[supply, supply, supply],
            BUFFER=[buffer, buffer, buffer],
            SHOCK_PROB=shock_prob
        )
        
        count_total = 0
        count_type_a = 0
        count_type_b = 0
        
        # 乱数シードのベース (パラメータごとに変えない方がノイズが減る)
        seed_base = 1000
        
        for r_a, r_b in target_profiles:
            vals_a = true_matrices[r_a]
            vals_b = true_matrices[r_b]
            report_truth = np.stack([vals_a, vals_b])
            
            # 1. 正直申告のEU (モンテカルロ)
            eu_truth_a = calculate_eu_monte_carlo(cfg, report_truth, vals_a, vals_b, MC_SAMPLES, seed_base)
            
            max_eu_lie = -float('inf')
            best_lie_idx = -1
            
            # 2. 嘘の最大探索
            for idx_lie, vals_lie in enumerate(strategy_matrices):
                report_lie = np.stack([vals_lie, vals_b])
                
                # 同じシードセットを使って比較する (分散低減法)
                eu_lie_a = calculate_eu_monte_carlo(cfg, report_lie, vals_a, vals_b, MC_SAMPLES, seed_base)
                
                if eu_lie_a > max_eu_lie:
                    max_eu_lie = eu_lie_a
                    best_lie_idx = idx_lie
            
            if max_eu_lie - eu_truth_a > 1e-7:
                count_total += 1
                true_ranks = true_types[r_a]['ranks']
                lie_ranks = strategy_types[best_lie_idx]['ranks']
                v_type = classify_lie_type(true_ranks, lie_ranks)
                
                if v_type == "Type A": count_type_a += 1
                elif v_type == "Type B": count_type_b += 1
        
        results.append({
            "parameter": val,
            "violation_total": count_total,
            "violation_type_a": count_type_a,
            "violation_type_b": count_type_b
        })
    return pd.DataFrame(results)

def plot_results(df, x_label, title, filename, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(df["parameter"], df["violation_total"], marker='o', label="Total", color='black', linewidth=2)
    plt.plot(df["parameter"], df["violation_type_a"], marker='x', label="Type A (Tie-Break)", color='blue', linestyle='--')
    plt.plot(df["parameter"], df["violation_type_b"], marker='^', label="Type B (Structure)", color='red', linestyle='--')
    
    plt.title(title + f" ({STRATEGY_MODE})")
    plt.xlabel(x_label)
    plt.ylabel(f"Violations (out of {SAMPLE_SIZE})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results_comparative", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results to: {output_dir}")
    print(f"Strategy Mode: {STRATEGY_MODE}")
    print(f"Method: Monte Carlo (Samples={MC_SAMPLES})")

    # 1. 選好生成
    print("Generating types...")
    true_types = generate_consistent_with_truncation(2, 3)
    true_matrices = [ranks_to_matrix(pt['ranks']) for pt in true_types]
    
    if STRATEGY_MODE == "UNRESTRICTED":
        strategy_types = generate_all_permutations(2, 3)
    elif STRATEGY_MODE == "UNRESTRICTED_WEAK":
        print("Warning: Generating Weak Orders...")
        strategy_types = generate_all_weak_orders(2, 3)
    else:
        strategy_types = true_types 
        
    strategy_matrices = [ranks_to_matrix(pt['ranks']) for pt in strategy_types]
    print(f"Truth: {len(true_types)}, Strategy: {len(strategy_types)}")

    # サンプリング
    random.seed(42)
    all_truth_indices = list(range(len(true_types)))
    all_pairs = list(itertools.product(all_truth_indices, repeat=2))
    
    if len(all_pairs) > SAMPLE_SIZE:
        target_profiles = random.sample(all_pairs, SAMPLE_SIZE)
    else:
        target_profiles = all_pairs
        
    print(f"Target Profiles: {len(target_profiles)} pairs")

    # Experiments
    supply_range = [1, 5, 10, 15, 20]
    df = run_experiment_scenario(true_types, strategy_types, true_matrices, strategy_matrices, target_profiles, "Supply", supply_range)
    df.to_csv(os.path.join(output_dir, "exp1_supply.csv"), index=False)
    plot_results(df, "Supply", "Effect of Supply", "exp1_supply.png", output_dir)

    buffer_range = [0, 2, 4, 6, 8, 10]
    df = run_experiment_scenario(true_types, strategy_types, true_matrices, strategy_matrices, target_profiles, "Buffer", buffer_range)
    df.to_csv(os.path.join(output_dir, "exp2_buffer.csv"), index=False)
    plot_results(df, "Buffer", "Effect of Buffer", "exp2_buffer.png", output_dir)

    shock_range = [0, 2, 4, 6, 8, 10]
    df = run_experiment_scenario(true_types, strategy_types, true_matrices, strategy_matrices, target_profiles, "Shock", shock_range)
    df.to_csv(os.path.join(output_dir, "exp3_shock.csv"), index=False)
    plot_results(df, "Shock", "Effect of Shock", "exp3_shock.png", output_dir)

    print("Done.")

if __name__ == "__main__":
    main()