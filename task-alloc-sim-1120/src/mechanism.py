import numpy as np
import random
from collections import deque

def run_online_mechanism(cfg, report_vals, rng):
    """
    オンラインRSD + TTC調整 (アウトサイドオプション対応版)
    
    修正点:
    - val > 0 (正の効用) がある場合のみスロットを選択・移動する。
    - 効用が0のスロットしかない場合は、空きがあっても「辞退(配分なし)」を選ぶ。
    """
    allocation = np.zeros((cfg.A, cfg.M, cfg.T), dtype=int)
    remaining_demand = list(cfg.Q) 
    
    shocks = np.zeros(cfg.T, dtype=int)
    for t in range(cfg.T):
        r = rng.random()
        cum_p = 0
        for val, p in cfg.SHOCK_PROB.items():
            cum_p += p
            if r <= cum_p:
                shocks[t] = val
                break
    
    # --- 選好情報の事前処理 ---
    agent_ranked_slots = []
    max_rank_limit = 0
    
    for ag in range(cfg.A):
        slots = []
        for t in range(cfg.T):
            for m in range(cfg.M):
                slots.append(((m, t), report_vals[ag, m, t]))
        slots.sort(key=lambda x: x[1], reverse=True)
        
        ranked = []
        curr_rank = 0
        if slots:
            ranked.append((slots[0][0], slots[0][1], 0))
            for i in range(1, len(slots)):
                if slots[i][1] < slots[i-1][1]: curr_rank += 1
                ranked.append((slots[i][0], slots[i][1], curr_rank))
        agent_ranked_slots.append(ranked)
        if curr_rank > max_rank_limit:
            max_rank_limit = curr_rank

    def get_rank(agent, slot):
        for s, v, r in agent_ranked_slots[agent]:
            if s == slot: return r
        return 9999

    def has_valid_move(agent, current_slot, rank_drop, t_now, t_limit):
        cur_r = get_rank(agent, current_slot)
        threshold = cur_r + rank_drop
        for s, v, r in agent_ranked_slots[agent]:
            if r > threshold: continue 
            if t_now <= s[1] < t_limit:
                # ★追加: 効用が0より大きい（辞退ではない）場合のみ有効
                if v > 0: 
                    return True
        return False

    def get_targets_within_rank_drop(agent, current_slot, rank_drop):
        cur_r = get_rank(agent, current_slot)
        threshold = cur_r + rank_drop
        targets = []
        for s, v, r in agent_ranked_slots[agent]:
            # ★追加: 効用正のみ対象
            if r <= threshold and v > 0: 
                targets.append(s)
        return targets

    # --- 時間進行ループ ---
    for t in range(cfg.T):
        
        # ==========================================
        # Phase A: Window RSD
        # ==========================================
        window_end = min(cfg.T, t + cfg.WINDOW)
        agents_order = list(range(cfg.A))
        rng.shuffle(agents_order)
        
        for agent in agents_order:
            while remaining_demand[agent] > 0:
                best_slot = None
                best_val = 0 # ★修正: 基準値を -inf から 0 に変更 (0より大きいものだけ探す)
                
                for w_t in range(t, window_end):
                    safe_cap = max(0, int(cfg.BASE_SUPPLY[w_t] - cfg.BUFFER[w_t]))
                    if allocation[:, :, w_t].sum() < safe_cap:
                        for m in range(cfg.M):
                            if allocation[:, m, w_t].sum() < cfg.MACHINE_CAPACITY:
                                val = report_vals[agent, m, w_t]
                                # ★修正: val > 0 の条件は best_val=0 で担保
                                if val > best_val:
                                    best_val = val
                                    best_slot = (m, w_t)
                
                if best_slot:
                    allocation[agent, best_slot[0], best_slot[1]] += 1
                    remaining_demand[agent] -= 1
                else:
                    # 取れるスロットがない、または全ての空きスロットが「価値0(辞退)」
                    break

        # ==========================================
        # Phase B: Execution & Adjustment (TTC)
        # ==========================================
        power_supply = max(0, int(cfg.BASE_SUPPLY[t] + shocks[t]))
        real_cap = min(power_supply, cfg.M * cfg.MACHINE_CAPACITY)
        
        current_usage = allocation[:, :, t].sum()
        
        # --- Case 1: 供給不足 (Eviction Chain) ---
        if current_usage > real_cap:
            deficit = current_usage - real_cap
            search_limit = max_rank_limit + 2
            
            for rank_drop in range(search_limit): 
                if deficit <= 0: break
                
                while deficit > 0:
                    start_nodes = []
                    for ag in range(cfg.A):
                        for m in range(cfg.M):
                            if allocation[ag, m, t] > 0:
                                if has_valid_move(ag, (m, t), rank_drop, t, cfg.T):
                                    start_nodes.append((ag, m, t))
                    
                    start_nodes = list(set(start_nodes))
                    rng.shuffle(start_nodes)
                    if not start_nodes: break 

                    queue = deque()
                    visited = set()
                    
                    for (ag, m, time_idx) in start_nodes:
                        state = (ag, (m, time_idx))
                        if state not in visited:
                            queue.append((state, [])) 
                            visited.add(state)
                    
                    found_path = None
                    
                    while queue:
                        (curr_ag, curr_slot), history = queue.popleft()
                        
                        # ★ここでは get_targets_within_rank_drop 内で v>0 フィルタ済み
                        targets = get_targets_within_rank_drop(curr_ag, curr_slot, rank_drop)
                        rng.shuffle(targets)
                        
                        sinks = []
                        relays = []
                        
                        for tgt_slot in targets:
                            tgt_m, tgt_t = tgt_slot
                            if not (t <= tgt_t < cfg.T): continue
                            
                            if tgt_t == t: p_limit = real_cap
                            else: p_limit = max(0, int(cfg.BASE_SUPPLY[tgt_t] - cfg.BUFFER[tgt_t]))
                            
                            is_p_ok = (allocation[:, :, tgt_t].sum() < p_limit)
                            is_m_ok = (allocation[:, tgt_m, tgt_t].sum() < cfg.MACHINE_CAPACITY)
                            
                            if is_p_ok and is_m_ok: sinks.append(tgt_slot)
                            else: relays.append((tgt_slot, is_m_ok, is_p_ok))

                        if sinks:
                            tgt_slot = sinks[0]
                            found_path = history + [(curr_ag, curr_slot, tgt_slot)]
                            break
                        
                        for (tgt_slot, is_m_ok, is_p_ok) in relays:
                            tgt_m, tgt_t = tgt_slot
                            possible_victims = []
                            if not is_m_ok: 
                                for v_ag in range(cfg.A):
                                    if allocation[v_ag, tgt_m, tgt_t] > 0:
                                        possible_victims.append((v_ag, tgt_m))
                            elif is_m_ok and not is_p_ok:
                                for v_ag in range(cfg.A):
                                    for v_m in range(cfg.M):
                                        if allocation[v_ag, v_m, tgt_t] > 0:
                                            possible_victims.append((v_ag, v_m))
                            
                            if not possible_victims: continue
                            valid_victims = []
                            for v_ag, v_m in possible_victims:
                                v_slot = (v_m, tgt_t)
                                if (v_ag, v_slot) == (curr_ag, curr_slot): continue
                                if has_valid_move(v_ag, v_slot, rank_drop, t, cfg.T):
                                    valid_victims.append((v_ag, v_slot))
                            
                            if valid_victims:
                                v_ag, v_slot = rng.choice(valid_victims)
                                next_state = (v_ag, v_slot)
                                if next_state not in visited:
                                    visited.add(next_state)
                                    new_hist = history + [(curr_ag, curr_slot, tgt_slot)]
                                    queue.append((next_state, new_hist))
                        
                        if found_path: break
                    
                    if found_path:
                        first_ag, first_src, _ = found_path[0]
                        allocation[first_ag, first_src[0], first_src[1]] -= 1
                        for i, (ag, src, dst) in enumerate(found_path):
                            allocation[ag, dst[0], dst[1]] += 1
                            if i < len(found_path) - 1:
                                next_ag, next_src, _ = found_path[i+1]
                                allocation[next_ag, next_src[0], next_src[1]] -= 1
                        deficit -= 1
                    else:
                        # 削除確定
                        victims = []
                        for ag in range(cfg.A):
                            for m in range(cfg.M):
                                if allocation[ag, m, t] > 0: victims.append((ag, m))
                        if victims:
                            v_ag, v_m = rng.choice(victims)
                            allocation[v_ag, v_m, t] -= 1
                            remaining_demand[v_ag] += 1
                            deficit -= 1
                        else: deficit = 0
                        break

        # --- Case 2: 供給過多 (Promotion) ---
        elif current_usage < real_cap:
            agents_for_promo = list(range(cfg.A))
            rng.shuffle(agents_for_promo)
            
            filled_something = True
            while filled_something and allocation[:, :, t].sum() < real_cap:
                filled_something = False
                for ag in agents_for_promo:
                    if remaining_demand[ag] <= 0: continue
                    if allocation[:, :, t].sum() >= real_cap: break
                    
                    best_m = None
                    best_val = 0 # ★修正: 0より大きいものだけ (辞退考慮)
                    
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

    return allocation