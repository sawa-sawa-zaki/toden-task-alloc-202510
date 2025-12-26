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
from src.mechanism import run_online_mechanism

# --- ★高速化のための設定変更 ---
SAMPLE_SIZE = 1000         # 1. 全ペア見ない (46656 -> 200)
LIE_SAMPLE_SIZE = 216      # 2. 嘘も全通り試さない (216 -> 50)
MC_SAMPLES = 20          # 3. 試行回数を減らす (30 -> 15)

DEFAULT_DEMAND = [20, 20]
DEFAULT_SUPPLY = 10
DEFAULT_BUFFER = 2
DEFAULT_SHOCK_MAG = 5
STRATEGY_MODE = "RESTRICTED"

def calculate_eu_monte_carlo(cfg, report_vals, true_vals_a, true_vals_b, samples, seed_base):
    total_u_a = 0.0
    for i in range(samples):
        rng = random.Random(seed_base + i)
        alloc = run_online_mechanism(cfg, report_vals, rng)
        u_a = np.sum(alloc[0] * true_vals_a)
        total_u_a += u_a
    return total_u_a / samples

def run_experiment_scenario(true_types, strategy_types, true_matrices, strategy_matrices, target_profiles, variable_name, range_values):
    results = []
    print(f"--- Experiment: Varying {variable_name} (Mode: {STRATEGY_MODE}, N={SAMPLE_SIZE}) ---")
    
    # 嘘の探索候補を事前にサンプリング
    all_lie_indices = list(range(len(strategy_matrices)))
    if len(all_lie_indices) > LIE_SAMPLE_SIZE:
        sampled_lie_indices = random.sample(all_lie_indices, LIE_SAMPLE_SIZE)
    else:
        sampled_lie_indices = all_lie_indices
        
    print(f"  Lie Search Space: {len(sampled_lie_indices)} / {len(strategy_matrices)}")

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
        
        # ★追加: 効用の合計を記録するための変数
        total_utility_sum = 0.0
        
        seed_base = 1000
        
        for r_a, r_b in target_profiles:
            vals_a = true_matrices[r_a]
            vals_b = true_matrices[r_b]
            report_truth = np.stack([vals_a, vals_b])
            
            # 1. 正直申告時のEU (これを社会厚生のプロキシとする)
            eu_truth_a = calculate_eu_monte_carlo(cfg, report_truth, vals_a, vals_b, MC_SAMPLES, seed_base)
            
            # 統計用に加算 (平均を取るため)
            total_utility_sum += eu_truth_a
            
            max_eu_lie = -float('inf')
            best_lie_idx = -1
            
            # 2. 嘘の探索
            for idx_lie in sampled_lie_indices:
                vals_lie = strategy_matrices[idx_lie]
                report_lie = np.stack([vals_lie, vals_b])
                
                eu_lie_a = calculate_eu_monte_carlo(cfg, report_lie, vals_a, vals_b, MC_SAMPLES, seed_base)
                
                if eu_lie_a > max_eu_lie:
                    max_eu_lie = eu_lie_a
                    best_lie_idx = idx_lie
            
            # 判定
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
            "violation_type_b": count_type_b,
            "avg_utility": total_utility_sum / len(target_profiles) # ★平均効用
        })
        
    return pd.DataFrame(results)

def plot_results(df, x_label, title, filename, output_dir):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左軸: 違反数 (棒グラフや線グラフ)
    color = 'tab:red'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(f"Violations (out of {SAMPLE_SIZE})", color=color)
    ax1.plot(df["parameter"], df["violation_total"], marker='o', label="Total Violations", color=color, linewidth=2)
    # Type A/B の内訳も薄く表示
    ax1.plot(df["parameter"], df["violation_type_b"], marker='^', label="Type B (Severe)", color='salmon', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # 右軸: 平均効用 (効率性)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Average Honest Utility', color=color)  
    ax2.plot(df["parameter"], df["avg_utility"], marker='s', label="Avg Utility", color=color, linewidth=2, linestyle='-.')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results_comparative_fast", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results to: {output_dir}")
    print(f"Config: Profiles={SAMPLE_SIZE}, LieSearch={LIE_SAMPLE_SIZE}, MC={MC_SAMPLES}")

    # 1. 選好生成
    print("Generating types...")
    true_types = generate_consistent_with_truncation(2, 3)
    true_matrices = [ranks_to_matrix(pt['ranks']) for pt in true_types]
    
    if STRATEGY_MODE == "UNRESTRICTED":
        strategy_types = generate_all_permutations(2, 3)
    elif STRATEGY_MODE == "UNRESTRICTED_WEAK":
        strategy_types = generate_all_weak_orders(2, 3)
    else:
        strategy_types = true_types 
        
    strategy_matrices = [ranks_to_matrix(pt['ranks']) for pt in strategy_types]
    print(f"Truth: {len(true_types)}, Strategy: {len(strategy_types)}")

    # 2. サンプリング
    random.seed(42)
    all_truth_indices = list(range(len(true_types)))
    all_pairs = list(itertools.product(all_truth_indices, repeat=2))
    
    if len(all_pairs) > SAMPLE_SIZE:
        target_profiles = random.sample(all_pairs, SAMPLE_SIZE)
        print(f"Target Profiles: {len(target_profiles)} (Sampled from {len(all_pairs)})")
    else:
        target_profiles = all_pairs
        print(f"Target Profiles: {len(target_profiles)} (All)")

    # --- Experiments ---
    
    # 1. Supply
    supply_range = [1, 5, 10, 15, 20]
    df = run_experiment_scenario(
        true_types, strategy_types, true_matrices, strategy_matrices, 
        target_profiles, "Supply", supply_range
    )
    df.to_csv(os.path.join(output_dir, "exp1_supply.csv"), index=False)
    plot_results(df, "Supply", "Effect of Supply on IC & Efficiency", "exp1_supply.png", output_dir)

    # 2. Buffer
    buffer_range = [0, 1, 2, 3, 4, 5, 6, 8, 10]
    df = run_experiment_scenario(
        true_types, strategy_types, true_matrices, strategy_matrices, 
        target_profiles, "Buffer", buffer_range
    )
    df.to_csv(os.path.join(output_dir, "exp2_buffer.csv"), index=False)
    plot_results(df, "Buffer", "Effect of Buffer on IC & Efficiency", "exp2_buffer.png", output_dir)

    # 3. Shock
    shock_range = [0, 2, 4, 6, 8, 10]
    df = run_experiment_scenario(
        true_types, strategy_types, true_matrices, strategy_matrices, 
        target_profiles, "Shock", shock_range
    )
    df.to_csv(os.path.join(output_dir, "exp3_shock.csv"), index=False)
    plot_results(df, "Shock", "Effect of Shock on IC & Efficiency", "exp3_shock.png", output_dir)

    print("Done.")

if __name__ == "__main__":
    main()