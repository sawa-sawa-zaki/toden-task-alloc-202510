import numpy as np
import random

def run_online_mechanism(cfg, report_vals, rng):
    """
    オンライン(逐次)でのRSD + TTC実行 (修正完全版)
    
    修正点:
    1. 供給過多時(Positive Shock)の「繰り上げ(Promotion)」処理を追加。
       - 同時同量原則に基づき、物理キャパシティと需要がある限り埋める。
    2. 供給不足時(Negative Shock)の「無差別スライド優先」処理を強化。
       - 追い出し発生時、効用が下がらない(無差別な)移動先を最優先で探す。
    """
    # 初期化
    allocation = np.zeros((cfg.A, cfg.M, cfg.T), dtype=int)
    remaining_demand = list(cfg.Q) 
    
    # ショック生成
    shocks = np.zeros(cfg.T, dtype=int)
    for t in range(cfg.T):
        r = rng.random()
        cum_p = 0
        for val, p in cfg.SHOCK_PROB.items():
            cum_p += p
            if r <= cum_p:
                shocks[t] = val
                break
    
    # --- 時間進行ループ ---
    for t in range(cfg.T):
        
        # ==========================================
        # Phase A: Window RSD (事前配分)
        # ==========================================
        window_end = min(cfg.T, t + cfg.WINDOW)
        
        # バッファを考慮したRSD
        # エージェントをシャッフルして順番に「安全枠」を埋めていく
        agents_order = list(range(cfg.A))
        rng.shuffle(agents_order)
        
        for agent in agents_order:
            # 自分の需要が尽きていないか、すでにこの時間帯tで枠を持っていないか確認
            # (※モデルによっては1人が同時刻に複数マシン使用可だが、ここでは需要分だけループ)
            while remaining_demand[agent] > 0:
                
                best_slot = None
                best_val = -float('inf')
                
                # ウィンドウ内を探索
                for w_t in range(t, window_end):
                    # 安全キャパ = 予測供給 - バッファ
                    safe_cap = max(0, int(cfg.BASE_SUPPLY[w_t] - cfg.BUFFER[w_t]))
                    
                    # その時間帯全体の負荷チェック
                    if allocation[:, :, w_t].sum() < safe_cap:
                        # 空いているマシンを探す
                        for m in range(cfg.M):
                            if allocation[:, m, w_t].sum() == 0: # 誰も使ってない
                                val = report_vals[agent, m, w_t]
                                if val > best_val:
                                    best_val = val
                                    best_slot = (m, w_t)
                
                # ベストな枠が見つかったら確保
                if best_slot:
                    # すでに自分が確保しているスロットならスキップ(二重取り防止)などの制御も必要だが
                    # allocation[:, m, w_t].sum() == 0 で弾いているのでOK
                    allocation[agent, best_slot[0], best_slot[1]] = 1
                    remaining_demand[agent] -= 1
                else:
                    # このターンではもう取れるものがない
                    break

        # ==========================================
        # Phase B: Execution & Adjustment (実需給調整)
        # ==========================================
        
        # 実供給キャパシティ = 予測 + ショック
        # ただし、物理的なマシンの台数(M)を超えることはできない
        power_cap = max(0, int(cfg.BASE_SUPPLY[t] + shocks[t]))
        real_cap = min(power_cap, cfg.M) 
        
        # 現在 t に割り当てられているタスク
        users_at_t = [] # (agent, machine)
        for ag in range(cfg.A):
            for m in range(cfg.M):
                if allocation[ag, m, t] == 1:
                    users_at_t.append((ag, m))
        
        current_usage = len(users_at_t)
        
        # ------------------------------------------
        # Case 1: 供給不足 (Eviction & Slide)
        # ------------------------------------------
        if current_usage > real_cap:
            num_evict = current_usage - real_cap
            
            # ランダムに追い出し対象を決定
            rng.shuffle(users_at_t)
            evicted = users_at_t[:num_evict]
            
            for ag, m in evicted:
                # 1. 割当解除
                allocation[ag, m, t] = 0
                remaining_demand[ag] += 1
                original_val = report_vals[ag, m, t]
                
                # 2. 再配分 (Chain Move) - 無差別優先
                # 探索範囲: 現在〜ウィンドウエンドの空き地
                
                candidates = []
                
                for w_t_alt in range(t, window_end):
                    # 移動先の許容キャパ
                    # t (現在) -> real_cap
                    # t+k (未来) -> safe_cap (まだショック不明なためバッファ内)
                    if w_t_alt == t:
                        limit = real_cap
                    else:
                        limit = max(0, int(cfg.BASE_SUPPLY[w_t_alt] - cfg.BUFFER[w_t_alt]))
                    
                    # キャパ空きあり？
                    if allocation[:, :, w_t_alt].sum() < limit:
                        for m_alt in range(cfg.M):
                            if allocation[:, m_alt, w_t_alt].sum() == 0:
                                val = report_vals[ag, m_alt, w_t_alt]
                                candidates.append({
                                    "slot": (m_alt, w_t_alt),
                                    "val": val,
                                    "is_indifferent": (val >= original_val) # 効用が下がらないか
                                    # ※RSDでベストを取っているはずなので、基本は val <= original_val
                                    #   つまり val == original_val が「無差別」
                                })
                
                if not candidates:
                    continue # 救済不可
                
                # 選定ロジック:
                # 1. 「無差別(効用低下なし)」の候補があるか？あればその中でベストを選ぶ
                # 2. なければ、「効用低下するがマシなもの」の中でベストを選ぶ
                
                indifferent_moves = [c for c in candidates if c["is_indifferent"]]
                
                if indifferent_moves:
                    # 無差別候補の中から選ぶ (複数あるなら効用最大、まあ同じ値のはずだが)
                    best_move = max(indifferent_moves, key=lambda x: x["val"])
                else:
                    # 妥協候補の中からベストを選ぶ
                    best_move = max(candidates, key=lambda x: x["val"])
                
                # 移動実行
                new_m, new_t = best_move["slot"]
                allocation[ag, new_m, new_t] = 1
                remaining_demand[ag] -= 1

        # ------------------------------------------
        # Case 2: 供給過多 (Promotion / Backfilling)
        # ------------------------------------------
        elif current_usage < real_cap:
            # 空き枠がある場合、待機需要があるエージェントを繰り上げ当選させる
            # 同時同量原則に従い、可能な限り埋める
            
            # ランダムな順序でオファー (公平性のためシャッフル)
            # ※本来は「過去に我慢した人」優先などが望ましいが、ここではランダム
            agents_for_promo = list(range(cfg.A))
            rng.shuffle(agents_for_promo)
            
            # キャパが埋まるか、全員の需要が尽きるまでループ
            # 1巡だけでなく、埋まりきるまで繰り返す
            filled_something = True
            while filled_something and allocation[:, :, t].sum() < real_cap:
                filled_something = False
                
                for ag in agents_for_promo:
                    if remaining_demand[ag] <= 0: continue
                    if allocation[:, :, t].sum() >= real_cap: break
                    
                    # 時刻 t の空いているマシンを探す
                    best_m = None
                    best_val = -float('inf')
                    
                    for m in range(cfg.M):
                        if allocation[:, m, t].sum() == 0: # 空いている
                            val = report_vals[ag, m, t]
                            if val > best_val:
                                best_val = val
                                best_m = m
                    
                    if best_m is not None:
                        # 繰り上げ割当実行
                        allocation[ag, best_m, t] = 1
                        remaining_demand[ag] -= 1
                        filled_something = True

    return allocation