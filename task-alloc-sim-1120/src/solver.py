import numpy as np
import itertools
from src.config import Config

def get_exact_expected_utility(cfg: Config, report_vals, true_vals_a, true_vals_b):
    """
    厳密な期待効用(Expected Utility)を計算する。
    全ショックパターン × 全順列パターンの加重平均。
    """
    shock_options = list(cfg.SHOCK_PROB.keys())
    all_shock_patterns = list(itertools.product(shock_options, repeat=cfg.T))
    
    total_util_a = 0.0
    total_util_b = 0.0
    total_prob_weight = 0.0
    
    for shock_seq in all_shock_patterns:
        prob_weight = 1.0
        for s in shock_seq:
            prob_weight *= cfg.SHOCK_PROB[s]
        if prob_weight == 0: continue

        alloc = _run_deterministic_mechanism_averaged(cfg, report_vals, shock_seq)
        
        u_a = np.sum(alloc[0] * true_vals_a)
        u_b = np.sum(alloc[1] * true_vals_b)
        
        total_util_a += u_a * prob_weight
        total_util_b += u_b * prob_weight
        total_prob_weight += prob_weight
        
    return total_util_a, total_util_b

def _run_deterministic_mechanism_averaged(cfg, report_vals, shock_seq):
    """
    優先順位 [0,1] と [1,0] の結果を平均する
    """
    orders = [[0, 1], [1, 0]]
    sum_alloc = np.zeros((cfg.A, cfg.M, cfg.T))
    for order in orders:
        alloc = _run_fixed_order_allocation(cfg, report_vals, shock_seq, order)
        sum_alloc += alloc
    return sum_alloc / len(orders)

def _run_fixed_order_allocation(cfg, report_vals, shock_seq, priority_order):
    """
    固定された順序とショックに基づく決定論的配分
    """
    alloc = np.zeros((cfg.A, cfg.M, cfg.T), dtype=int)
    remaining_demand = list(cfg.Q) 
    
    for t in range(cfg.T):
        # === Phase A ===
        window_end = min(cfg.T, t + cfg.WINDOW)
        
        for agent in priority_order:
            while remaining_demand[agent] > 0:
                best_slot = None
                best_val = -float('inf')
                for w_t in range(t, window_end):
                    safe_cap = max(0, int(cfg.BASE_SUPPLY[w_t] - cfg.BUFFER[w_t]))
                    if alloc[:, :, w_t].sum() < safe_cap:
                        for m in range(cfg.M):
                            if alloc[:, m, w_t].sum() < cfg.MACHINE_CAPACITY:
                                val = report_vals[agent, m, w_t]
                                if val > best_val:
                                    best_val = val
                                    best_slot = (m, w_t)
                if best_slot:
                    alloc[agent, best_slot[0], best_slot[1]] = 1
                    remaining_demand[agent] -= 1
                else:
                    break

        # === Phase B ===
        power_cap = max(0, int(cfg.BASE_SUPPLY[t] + shock_seq[t]))
        real_cap = min(power_cap, cfg.M * cfg.MACHINE_CAPACITY)
        
        users_at_t = []
        # 優先順位の「逆順」(弱い順)でリストアップして追い出し候補にする
        for agent in reversed(priority_order):
            for m in range(cfg.M):
                if alloc[agent, m, t] == 1:
                    users_at_t.append((agent, m))
        
        current_usage = len(users_at_t)
        
        # Eviction
        if current_usage > real_cap:
            num_evict = current_usage - real_cap
            evicted = users_at_t[:num_evict] # 弱い順に確定で追い出し
            
            for ag, m in evicted:
                alloc[ag, m, t] = 0
                remaining_demand[ag] += 1
                original_val = report_vals[ag, m, t]
                
                candidates = []
                for w_t_alt in range(t, window_end):
                    if w_t_alt == t: limit = real_cap
                    else: limit = max(0, int(cfg.BASE_SUPPLY[w_t_alt] - cfg.BUFFER[w_t_alt]))
                    
                    if alloc[:, :, w_t_alt].sum() < limit:
                        for m_alt in range(cfg.M):
                            if alloc[:, m_alt, w_t_alt].sum() < cfg.MACHINE_CAPACITY:
                                val = report_vals[ag, m_alt, w_t_alt]
                                candidates.append({
                                    "slot": (m_alt, w_t_alt),
                                    "val": val,
                                    "is_indifferent": (val >= original_val)
                                })
                
                if not candidates: continue
                indifferent_moves = [c for c in candidates if c["is_indifferent"]]
                if indifferent_moves:
                    best_move = max(indifferent_moves, key=lambda x: x["val"])
                else:
                    best_move = max(candidates, key=lambda x: x["val"])
                
                new_m, new_t = best_move["slot"]
                alloc[ag, new_m, new_t] = 1
                remaining_demand[ag] -= 1
                
        # Promotion
        elif current_usage < real_cap:
            # 優先順位順に埋める (FIFO)
            filled_something = True
            while filled_something and alloc[:, :, t].sum() < real_cap:
                filled_something = False
                for ag in priority_order:
                    if remaining_demand[ag] <= 0: continue
                    if alloc[:, :, t].sum() >= real_cap: break
                    
                    best_m = None
                    best_val = -float('inf')
                    for m in range(cfg.M):
                        if alloc[:, m, t].sum() < cfg.MACHINE_CAPACITY:
                            val = report_vals[ag, m, t]
                            if val > best_val:
                                best_val = val
                                best_m = m
                    
                    if best_m is not None:
                        alloc[ag, best_m, t] = 1
                        remaining_demand[ag] -= 1
                        filled_something = True
                        
    return alloc