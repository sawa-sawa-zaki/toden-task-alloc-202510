import os
import sys
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime
import dataclasses

# srcフォルダからインポート
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import Config, ICConfig
from src.generator import (
    generate_consistent_with_truncation,
    generate_all_permutations, 
    generate_all_weak_orders, 
    ranks_to_matrix,
    UNACCEPTABLE,
    classify_lie_type
)
from src.mechanism import run_online_mechanism

# --- Helper: ランクを文字列にする関数 ---
def ranks_to_string(ranks, T=3, M=2):
    slots = []
    idx = 0
    machines = ["A", "B"]
    for t in range(T):
        for m in range(M):
            slot_name = f"{machines[m]}{t}"
            rank = ranks[idx]
            slots.append((rank, slot_name))
            idx += 1
    
    slots.sort(key=lambda x: x[0])
    
    res = ""
    first = True
    valid_slots = [s for s in slots if s[0] != UNACCEPTABLE]
    
    for i, (r, name) in enumerate(valid_slots):
        if i > 0:
            prev_r = valid_slots[i-1][0]
            if r == prev_r: res += "="
            else: res += ">"
        res += name
        first = False
    
    rejected = [name for r, name in slots if r == UNACCEPTABLE]
    if rejected:
        if not first:
            res += " >> "
        res += f"[Reject: {','.join(rejected)}]"
        
    return res

def main():
    cfg = Config()
    ic_cfg = ICConfig()
    
    # --- 1. 実行ごとのフォルダ作成 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(ic_cfg.OUTPUT_BASE_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Simulation Start ===")
    print(f"Output Directory: {output_dir}")
    print(f"Strategy Domain: {cfg.STRATEGY_DOMAIN}")

    # --- 2. Config保存 ---
    config_log_path = os.path.join(output_dir, "config_log.txt")
    with open(config_log_path, "w") as f:
        f.write(f"Timestamp: {timestamp}\n")
        for field in dataclasses.fields(cfg):
            f.write(f"{field.name}: {getattr(cfg, field.name)}\n")
        for field in dataclasses.fields(ic_cfg):
            f.write(f"{field.name}: {getattr(ic_cfg, field.name)}\n")

    # --- 3. 選好タイプの生成 ---
    
    # A. 真の選好
    print("Generating TRUE types (Consistent with Truncation)...")
    truth_types = generate_consistent_with_truncation(cfg.M, cfg.T)
    
    n_truth = len(truth_types)
    truth_matrices = [ranks_to_matrix(pt['ranks']) for pt in truth_types]
    print(f"Truth Types (Unique): {n_truth}")

    # B. 嘘の選好
    print(f"Generating STRATEGY types ({cfg.STRATEGY_DOMAIN})...")
    if cfg.STRATEGY_DOMAIN == "UNRESTRICTED":
        strategy_types = generate_all_permutations(cfg.M, cfg.T)
    elif cfg.STRATEGY_DOMAIN == "UNRESTRICTED_WEAK":
        strategy_types = generate_all_weak_orders(cfg.M, cfg.T)
    else:
        strategy_types = truth_types
        
    n_strategy = len(strategy_types)
    strategy_matrices = [ranks_to_matrix(pt['ranks']) for pt in strategy_types]
    print(f"Strategy Types: {n_strategy}")

    # --- 4. Outcome Matrix ---
    unique_matrices_A = []
    mat_to_unique_idx = {} 
    
    def register_matrix(mat):
        b = mat.tobytes()
        if b not in mat_to_unique_idx:
            mat_to_unique_idx[b] = len(unique_matrices_A)
            unique_matrices_A.append(mat)
        return mat_to_unique_idx[b]

    map_truth_to_unique = {}    
    map_strategy_to_unique = {} 

    for i, mat in enumerate(truth_matrices):
        map_truth_to_unique[i] = register_matrix(mat)
    for i, mat in enumerate(strategy_matrices):
        map_strategy_to_unique[i] = register_matrix(mat)
        
    n_unique_A = len(unique_matrices_A)
    n_cols_B = len(truth_matrices)
    
    total_calc = n_unique_A * n_cols_B
    print(f"Optimized Input A: {n_unique_A} unique types")
    print(f"Total Pairs to Simulate: {total_calc}")
    
    outcome_cache = {} 
    calc_pairs = list(itertools.product(range(n_unique_A), range(n_cols_B)))
    
    for u_idx_a, idx_b in tqdm(calc_pairs, desc="Simulating Outcomes"):
        mat_a = unique_matrices_A[u_idx_a]
        mat_b = truth_matrices[idx_b]
        
        report_vals = np.stack([mat_a, mat_b])
        avg_alloc = np.zeros((cfg.A, cfg.M, cfg.T))
        
        seeds = [ic_cfg.SEED + s for s in range(ic_cfg.SAMPLES)]
        for seed in seeds:
            rng = random.Random(seed)
            alloc = run_online_mechanism(cfg, report_vals, rng)
            avg_alloc += alloc
        avg_alloc /= ic_cfg.SAMPLES
        
        outcome_cache[(u_idx_a, idx_b)] = avg_alloc

    # --- 5. IC検証 ---
    print("Verifying IC...")
    summary_data = []
    count_type_a = 0
    count_type_b = 0
    
    truth_indices = list(range(n_truth))
    
    # ★修正: MAX_PROFILES が None の場合の安全対策を追加
    should_sample = False
    if ic_cfg.MAX_PROFILES is not None:
        if len(truth_indices)**2 > ic_cfg.MAX_PROFILES:
            should_sample = True
            
    if should_sample:
        random.seed(ic_cfg.SEED)
        target_pairs = []
        for _ in range(ic_cfg.MAX_PROFILES):
            target_pairs.append((random.choice(truth_indices), random.choice(truth_indices)))
        print(f"Analyzing {len(target_pairs)} profiles (Sampled from {len(truth_indices)**2})...")
    else:
        target_pairs = list(itertools.product(truth_indices, repeat=2))
        print(f"Analyzing ALL {len(target_pairs)} profiles...")
    
    for p_idx, (idx_true_a, idx_true_b) in tqdm(enumerate(target_pairs), total=len(target_pairs), desc="Analyzing"):
        
        u_key_truth = map_truth_to_unique[idx_true_a]
        alloc_truth = outcome_cache[(u_key_truth, idx_true_b)]
        val_matrix_a = truth_matrices[idx_true_a]
        u_truth = np.sum(alloc_truth[0] * val_matrix_a)
        
        max_u_lie = -1.0
        best_lie_idx = -1
        
        for idx_lie in range(n_strategy):
            u_key_lie = map_strategy_to_unique[idx_lie]
            alloc_lie = outcome_cache[(u_key_lie, idx_true_b)]
            u_lie = np.sum(alloc_lie[0] * val_matrix_a)
            
            if u_lie > max_u_lie:
                max_u_lie = u_lie
                best_lie_idx = idx_lie
        
        regret = max_u_lie - u_truth
        is_violation = (regret > 1e-5)
        
        violation_type = "None"
        if is_violation:
            true_ranks = truth_types[idx_true_a]['ranks']
            lie_ranks = strategy_types[best_lie_idx]['ranks']
            violation_type = classify_lie_type(true_ranks, lie_ranks)
            
            if violation_type == "Type A": count_type_a += 1
            elif violation_type == "Type B": count_type_b += 1
        
        true_str = ranks_to_string(truth_types[idx_true_a]['ranks'])
        lie_str = ranks_to_string(strategy_types[best_lie_idx]['ranks'])

        summary_data.append({
            "profile_idx": p_idx,
            "agent": "A",
            "true_type_idx": idx_true_a,
            "true_rank_str": true_str,
            "opp_type_idx": idx_true_b,
            "honest_utility": u_truth,
            "max_lie_utility": max_u_lie,
            "best_lie_idx": best_lie_idx,
            "best_lie_str": lie_str,
            "regret": regret,
            "is_violation": is_violation,
            "violation_type": violation_type
        })

    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "ic_full_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    
    df_violations = df_summary[df_summary["is_violation"] == True]
    violation_path = os.path.join(output_dir, "ic_violations_only.csv")
    df_violations.to_csv(violation_path, index=False)

    print(f"\n=== Finished ===")
    print(f"Truth Types: {n_truth}")
    print(f"Optimized Input A: {n_unique_A}")
    print(f"Total Violations: {len(df_violations)}")
    print(f"  - Type A (Tie-Break): {count_type_a}")
    print(f"  - Type B (Structural): {count_type_b}")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    main()