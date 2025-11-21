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
from src.generator import generate_time_monotone_prefs, ranks_to_matrix
from src.mechanism import run_online_mechanism

def main():
    cfg = Config()
    ic_cfg = ICConfig()
    
    # --- 1. 実行ごとのフォルダ作成 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(ic_cfg.OUTPUT_BASE_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Simulation Start ===")
    print(f"Output Directory: {output_dir}")

    # --- 2. 設定(Config)の保存 ---
    # どんな設定で回したか後で必ず確認できるようにする
    config_log_path = os.path.join(output_dir, "config_log.txt")
    with open(config_log_path, "w") as f:
        f.write("=== Simulation Configuration ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("\n[Market Config]\n")
        for field in dataclasses.fields(cfg):
            val = getattr(cfg, field.name)
            f.write(f"{field.name}: {val}\n")
        f.write("\n[IC Config]\n")
        for field in dataclasses.fields(ic_cfg):
            val = getattr(ic_cfg, field.name)
            f.write(f"{field.name}: {val}\n")
    print(f"Config saved to {config_log_path}")

    # --- 3. Outcome Matrix の構築 ---
    print("Generating preference types...")
    pref_types = generate_time_monotone_prefs(cfg.M, cfg.T)
    n_types = len(pref_types)
    print(f"Generated {n_types} types.")
    
    type_matrices = []
    for pt in pref_types:
        type_matrices.append(ranks_to_matrix(pt['ranks']))
    
    print(f"Pre-computing Outcome Matrix ({n_types}^2 combinations)...")
    outcome_cache = {} 
    input_pairs = list(itertools.product(range(n_types), repeat=cfg.A))
    
    # シミュレーション実行
    for r_a, r_b in tqdm(input_pairs, desc="Simulating Outcomes"):
        report_vals = np.stack([type_matrices[r_a], type_matrices[r_b]])
        avg_alloc = np.zeros((cfg.A, cfg.M, cfg.T))
        
        seeds = [ic_cfg.SEED + s for s in range(ic_cfg.SAMPLES)]
        for seed in seeds:
            rng = random.Random(seed)
            alloc = run_online_mechanism(cfg, report_vals, rng)
            avg_alloc += alloc
        avg_alloc /= ic_cfg.SAMPLES
        outcome_cache[(r_a, r_b)] = avg_alloc

    # --- 4. ICの全件検証 & ログ記録 ---
    print("Verifying IC & Recording Stats...")
    
    summary_data = []   # 全プロファイルのデータを格納
    violation_count = 0
    
    target_profiles = list(itertools.product(range(n_types), repeat=cfg.A))
    if ic_cfg.MAX_PROFILES:
        random.seed(ic_cfg.SEED)
        target_profiles = random.sample(target_profiles, ic_cfg.MAX_PROFILES)
        
    for p_idx, (true_a, true_b) in tqdm(enumerate(target_profiles), total=len(target_profiles), desc="Analyzing"):
        
        # === 結果の表引き ===
        alloc_truth = outcome_cache[(true_a, true_b)]
        
        # 正直申告時の効用（評価は常に真のタイプで行う）
        u_truth_a = np.sum(alloc_truth[0] * type_matrices[true_a])
        u_truth_b = np.sum(alloc_truth[1] * type_matrices[true_b])
        
        # --- Agent A の分析 ---
        max_u_lie_a = -1.0
        best_lie_a = -1
        
        # 全ての「嘘」を試して、最大利得を探す
        for lie_a in range(n_types):
            # 自分がlie_a, 相手は正直(true_b)
            alloc_lie = outcome_cache[(lie_a, true_b)]
            u_lie = np.sum(alloc_lie[0] * type_matrices[true_a])
            
            if u_lie > max_u_lie_a:
                max_u_lie_a = u_lie
                best_lie_a = lie_a
        
        regret_a = max_u_lie_a - u_truth_a
        is_violation_a = (regret_a > 1e-5) and (best_lie_a != true_a)
        if is_violation_a: violation_count += 1

        # 記録 (Agent A)
        summary_data.append({
            "profile_idx": p_idx,
            "agent": "A",
            "true_type": true_a,
            "opp_type": true_b,
            "honest_utility": u_truth_a,
            "max_lie_utility": max_u_lie_a,
            "best_lie_type": best_lie_a,
            "regret": regret_a,
            "is_violation": is_violation_a
        })

        # --- Agent B の分析 ---
        max_u_lie_b = -1.0
        best_lie_b = -1
        
        for lie_b in range(n_types):
            # 相手は正直(true_a), 自分がlie_b
            alloc_lie = outcome_cache[(true_a, lie_b)]
            u_lie = np.sum(alloc_lie[1] * type_matrices[true_b])
            
            if u_lie > max_u_lie_b:
                max_u_lie_b = u_lie
                best_lie_b = lie_b

        regret_b = max_u_lie_b - u_truth_b
        is_violation_b = (regret_b > 1e-5) and (best_lie_b != true_b)
        if is_violation_b: violation_count += 1

        # 記録 (Agent B)
        summary_data.append({
            "profile_idx": p_idx,
            "agent": "B",
            "true_type": true_b,
            "opp_type": true_a,
            "honest_utility": u_truth_b,
            "max_lie_utility": max_u_lie_b,
            "best_lie_type": best_lie_b,
            "regret": regret_b,
            "is_violation": is_violation_b
        })

    # --- 5. CSV出力 ---
    df_summary = pd.DataFrame(summary_data)
    
    # 全データ保存
    summary_path = os.path.join(output_dir, "ic_full_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    
    # 違反データのみ抽出保存
    df_violations = df_summary[df_summary["is_violation"] == True]
    violation_path = os.path.join(output_dir, "ic_violations_only.csv")
    df_violations.to_csv(violation_path, index=False)

    print(f"\n=== Analysis Finished ===")
    print(f"Total Profiles Checked: {len(target_profiles)}")
    print(f"Total Violation Cases: {len(df_violations)} (Agent-wise count)")
    print(f"Full Summary saved to: {summary_path}")
    print(f"Violations saved to: {violation_path}")

if __name__ == "__main__":
    main()