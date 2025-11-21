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
from src.config import Config
from src.generator import generate_time_monotone_prefs, ranks_to_matrix
from src.solver import get_exact_expected_utility

SAMPLE_SIZE = 100 
DEFAULT_DEMAND = [20, 20]
DEFAULT_SUPPLY = 10
DEFAULT_BUFFER = 5
DEFAULT_SHOCK_MAG = 5

def run_experiment_scenario(pref_types, type_matrices, target_profiles, variable_name, range_values):
    results = []
    print(f"--- Experiment: Varying {variable_name} ---")
    
    for val in tqdm(range_values):
        supply = DEFAULT_SUPPLY
        buffer = DEFAULT_BUFFER
        shock_mag = DEFAULT_SHOCK_MAG
        
        if variable_name == "Supply": supply = val
        elif variable_name == "Buffer": buffer = val
        elif variable_name == "Shock": shock_mag = val
            
        if shock_mag == 0: shock_prob = {0: 1.0}
        else: shock_prob = {-shock_mag: 0.25, 0: 0.5, shock_mag: 0.25}

        cfg = Config(
            A=2, M=2, T=3, WINDOW=2,
            MACHINE_CAPACITY=20, # 十分大きく
            Q=DEFAULT_DEMAND,
            BASE_SUPPLY=[supply, supply, supply],
            BUFFER=[buffer, buffer, buffer],
            SHOCK_PROB=shock_prob
        )
        
        violation_count = 0
        for r_a, r_b in target_profiles:
            vals_a = type_matrices[r_a]
            vals_b = type_matrices[r_b]
            report_truth = np.stack([vals_a, vals_b])
            
            eu_truth_a, _ = get_exact_expected_utility(cfg, report_truth, vals_a, vals_b)
            
            max_eu_lie = -float('inf')
            for lie_a in range(len(pref_types)):
                if lie_a == r_a: continue
                vals_lie = type_matrices[lie_a]
                report_lie = np.stack([vals_lie, vals_b])
                eu_lie_a, _ = get_exact_expected_utility(cfg, report_lie, vals_a, vals_b)
                if eu_lie_a > max_eu_lie: max_eu_lie = eu_lie_a
            
            if max_eu_lie - eu_truth_a > 1e-7:
                violation_count += 1
        
        results.append({
            "parameter": val,
            "violation_count": violation_count,
            "violation_rate": violation_count / len(target_profiles)
        })
    return pd.DataFrame(results)

def plot_results(df, x_label, title, filename, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(df["parameter"], df["violation_count"], marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(f"Violations (out of {SAMPLE_SIZE})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results_comparative", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results to: {output_dir}")

    print("Initializing...")
    pref_types = generate_time_monotone_prefs(2, 3)
    type_matrices = [ranks_to_matrix(pt['ranks']) for pt in pref_types]
    
    random.seed(42)
    all_pairs = list(itertools.product(range(len(pref_types)), repeat=2))
    target_profiles = random.sample(all_pairs, SAMPLE_SIZE)

    # Exp 1: Supply 1->20
    df = run_experiment_scenario(pref_types, type_matrices, target_profiles, "Supply", range(1, 21))
    df.to_csv(os.path.join(output_dir, "exp1_supply.csv"), index=False)
    plot_results(df, "Supply", "Effect of Supply", "exp1_supply.png", output_dir)

    # Exp 2: Buffer 0->10
    df = run_experiment_scenario(pref_types, type_matrices, target_profiles, "Buffer", range(0, 11))
    df.to_csv(os.path.join(output_dir, "exp2_buffer.csv"), index=False)
    plot_results(df, "Buffer", "Effect of Buffer", "exp2_buffer.png", output_dir)

    # Exp 3: Shock 0->10
    df = run_experiment_scenario(pref_types, type_matrices, target_profiles, "Shock", range(0, 11))
    df.to_csv(os.path.join(output_dir, "exp3_shock.csv"), index=False)
    plot_results(df, "Shock", "Effect of Shock", "exp3_shock.png", output_dir)

    print("Done.")

if __name__ == "__main__":
    main()