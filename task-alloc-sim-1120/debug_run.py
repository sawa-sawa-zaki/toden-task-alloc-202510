import os
import sys
import numpy as np
import random
import pandas as pd
from datetime import datetime

# srcフォルダのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import Config
from src.generator import generate_time_monotone_prefs, ranks_to_matrix
# mechanismはロジック参照用だが、デバッグ実行は下記の独自関数で行う

# --- ログ保存用のヘルパークラス ---
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- デバッグ表示用の特別版メカニズム関数 ---
def run_debug_mechanism(cfg, report_vals, rng):
    print("\n" + "="*60)
    print(f"🛠 MECHANISM DEBUG RUN START (T={cfg.T}, Window={cfg.WINDOW})")
    print("="*60)

    allocation = np.zeros((cfg.A, cfg.M, cfg.T), dtype=int)
    remaining_demand = list(cfg.Q) 
    
    # ショック生成と表示
    shocks = np.zeros(cfg.T, dtype=int)
    print("\n⚡ [Step 0] Shock Generation")
    for t in range(cfg.T):
        r = rng.random()
        cum_p = 0
        for val, p in cfg.SHOCK_PROB.items():
            cum_p += p
            if r <= cum_p:
                shocks[t] = val
                break
        # Base Supply + Shock
        expected = cfg.BASE_SUPPLY[t]
        actual = max(0, expected + shocks[t])
        print(f"  Time {t}: Shock = {shocks[t]:+d} (Supply: {expected} -> {actual})")

    # --- 時間進行ループ ---
    for t in range(cfg.T):
        print(f"\n🕒 [Time {t}] Processing...")
        
        # === Phase A: Window RSD ===
        window_end = min(cfg.T, t + cfg.WINDOW)
        print(f"  🔹 Phase A: Window RSD (Window: {t} ~ {window_end-1})")
        
        agents_order = list(range(cfg.A))
        rng.shuffle(agents_order)
        print(f"     Agent Priority Order: {agents_order}")
        
        for agent in agents_order:
            taken_count = 0
            while remaining_demand[agent] > 0:
                best_slot = None
                best_val = -float('inf')
                
                for w_t in range(t, window_end):
                    # 安全キャパ = 予測 - バッファ
                    safe_cap = max(0, int(cfg.BASE_SUPPLY[w_t] - cfg.BUFFER[w_t]))
                    
                    if allocation[:, :, w_t].sum() < safe_cap:
                        for m in range(cfg.M):
                            if allocation[:, m, w_t].sum() < cfg.MACHINE_CAPACITY:
                                val = report_vals[agent, m, w_t]
                                if val > best_val:
                                    best_val = val
                                    best_slot = (m, w_t)
                
                if best_slot:
                    allocation[agent, best_slot[0], best_slot[1]] += 1
                    remaining_demand[agent] -= 1
                    taken_count += 1
                else:
                    break
            print(f"     Agent {agent} took {taken_count} units. (Remaining Demand: {remaining_demand[agent]})")

        # === Phase B: Execution & Adjustment ===
        power_cap = max(0, int(cfg.BASE_SUPPLY[t] + shocks[t]))
        real_cap = min(power_cap, cfg.M * cfg.MACHINE_CAPACITY)
        print(f"  🔸 Phase B: Adjustment (Real Cap: {real_cap})")
        
        # ユニット展開
        units_at_t = []
        for ag in range(cfg.A):
            for m in range(cfg.M):
                count = allocation[ag, m, t]
                if count > 0:
                    units_at_t.extend([(ag, m)] * count)
        
        current_usage = len(units_at_t)
        print(f"     Current Usage at T={t}: {current_usage} / {real_cap}")
        
        # Case 1: 供給不足
        if current_usage > real_cap:
            num_evict = current_usage - real_cap
            print(f"     ⚠️ Eviction Triggered! Removing {num_evict} units...")
            rng.shuffle(units_at_t)
            evicted_units = units_at_t[:num_evict]
            
            for i, (ag, m) in enumerate(evicted_units):
                allocation[ag, m, t] -= 1
                remaining_demand[ag] += 1
                original_val = report_vals[ag, m, t]
                
                candidates = []
                for w_t_alt in range(t, window_end):
                    if w_t_alt == t: limit = real_cap
                    else: limit = max(0, int(cfg.BASE_SUPPLY[w_t_alt] - cfg.BUFFER[w_t_alt]))
                    
                    if allocation[:, :, w_t_alt].sum() < limit:
                        for m_alt in range(cfg.M):
                            if allocation[:, m_alt, w_t_alt].sum() < cfg.MACHINE_CAPACITY:
                                val = report_vals[ag, m_alt, w_t_alt]
                                candidates.append({
                                    "slot": (m_alt, w_t_alt),
                                    "val": val,
                                    "is_indifferent": (val >= original_val)
                                })
                
                move_result = "Failed (No space)"
                if candidates:
                    indifferent_moves = [c for c in candidates if c["is_indifferent"]]
                    if indifferent_moves:
                        best_move = max(indifferent_moves, key=lambda x: x["val"])
                        move_result = f"Moved to {best_move['slot']} (Indifferent)"
                    else:
                        best_move = max(candidates, key=lambda x: x["val"])
                        move_result = f"Moved to {best_move['slot']} (Compromise)"
                    
                    new_m, new_t = best_move["slot"]
                    allocation[ag, new_m, new_t] += 1
                    remaining_demand[ag] -= 1
                
                print(f"       Evicted Unit {i+1} (Agent {ag}): {move_result}")

        # Case 2: 供給過多
        elif current_usage < real_cap:
            print(f"     ✨ Promotion Triggered! Filling vacancies...")
            agents_for_promo = list(range(cfg.A))
            rng.shuffle(agents_for_promo)
            filled_count = 0
            
            filled_something = True
            while filled_something and allocation[:, :, t].sum() < real_cap:
                filled_something = False
                for ag in agents_for_promo:
                    if remaining_demand[ag] <= 0: continue
                    if allocation[:, :, t].sum() >= real_cap: break
                    
                    best_m = None
                    best_val = -float('inf')
                    for m in range(cfg.M):
                        if allocation[:, m, t].sum() < cfg.MACHINE_CAPACITY:
                            val = report_vals[ag, m, t]
                            if val > best_val:
                                best_val = val
                                best_m = m
                    
                    if best_m is not None:
                        allocation[ag, best_m, t] += 1
                        remaining_demand[ag] -= 1
                        filled_something = True
                        filled_count += 1
            print(f"       Promoted {filled_count} units.")
        else:
            print("     ✅ Supply/Demand Balanced. No changes.")

    return allocation

# --- メイン実行部 ---
def debug_main():
    # 1. ログ保存の準備
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results_debug", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "debug_log.txt")
    
    # 標準出力をファイルとコンソールの両方に切り替え
    sys.stdout = DualLogger(log_file)

    print(f"📝 Saving debug log to: {log_file}")

    # 2. Config設定 (デバッグ用に分かりやすい値)
    print("🔧 Setting up Config...")
    cfg = Config(
        A=2, M=2, T=3, WINDOW=2,
        MACHINE_CAPACITY=10, 
        Q=[5, 5],              # 需要5ずつ
        BASE_SUPPLY=[4, 4, 4], # 供給4
        BUFFER=[1, 1, 1],      # バッファ1 (RSDでの配分は 4-1=3 まで)
        SHOCK_PROB={-2: 0.3, 0: 0.4, 2: 0.3} 
    )
    print(f"  Supply: {cfg.BASE_SUPPLY}, Buffer: {cfg.BUFFER}, Demand: {cfg.Q}")

    # 3. 選好の生成と選択
    print("\n🎲 Generating Preferences...")
    pref_types = generate_time_monotone_prefs(cfg.M, cfg.T)
    
    random.seed(42) 
    p_idx_a, p_idx_b = random.sample(range(len(pref_types)), 2)
    
    vals_a = ranks_to_matrix(pref_types[p_idx_a]['ranks'])
    vals_b = ranks_to_matrix(pref_types[p_idx_b]['ranks'])
    
    # 選好の表示
    def print_pref(name, idx, ranks, vals):
        print(f"\n👤 {name} (Type ID: {idx})")
        print(f"  Ranks (A1, B1, A2, B2, A3, B3): {ranks}")
        print("  Value Matrix:")
        print(vals)

    print_pref("Agent A", p_idx_a, pref_types[p_idx_a]['ranks'], vals_a)
    print_pref("Agent B", p_idx_b, pref_types[p_idx_b]['ranks'], vals_b)

    report_vals = np.stack([vals_a, vals_b])

    # 4. メカニズム実行
    rng = random.Random(123) # 再現性確保
    final_alloc = run_debug_mechanism(cfg, report_vals, rng)

    # 5. 最終結果の表示
    print("\n" + "="*60)
    print("🏁 FINAL RESULTS")
    print("="*60)
    
    total_util_a = np.sum(final_alloc[0] * vals_a)
    total_util_b = np.sum(final_alloc[1] * vals_b)
    
    print(f"\n📦 Final Allocation (Agent A):\n{final_alloc[0]}")
    print(f"💰 Utility A: {total_util_a:.2f}")
    
    print(f"\n📦 Final Allocation (Agent B):\n{final_alloc[1]}")
    print(f"💰 Utility B: {total_util_b:.2f}")
    
    print("\nStats:")
    print(f"  Total Units Allocated: {final_alloc.sum()}")
    print(f"  Demand Fulfillment A: {final_alloc[0].sum()}/{cfg.Q[0]}")
    print(f"  Demand Fulfillment B: {final_alloc[1].sum()}/{cfg.Q[1]}")

if __name__ == "__main__":
    debug_main()