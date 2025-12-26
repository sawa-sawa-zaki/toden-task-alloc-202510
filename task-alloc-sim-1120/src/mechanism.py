import numpy as np
from collections import deque


def run_online_mechanism(cfg, report_vals, rng):
    allocation = np.zeros((cfg.A, cfg.M, cfg.T), dtype=int)
    remaining_demand = list(cfg.Q)

    # ---- shocks ----
    shocks = np.zeros(cfg.T, dtype=int)
    for t in range(cfg.T):
        r = rng.random()
        cum = 0
        for v, p in cfg.SHOCK_PROB.items():
            cum += p
            if r <= cum:
                shocks[t] = v
                break

    # ---- ranking ----
    agent_ranked_slots = []
    max_rank_limit = 0
    for ag in range(cfg.A):
        slots = []
        for t in range(cfg.T):
            for m in range(cfg.M):
                slots.append(((m, t), report_vals[ag, m, t]))
        slots.sort(key=lambda x: x[1], reverse=True)

        ranked = []
        r = 0
        ranked.append((slots[0][0], slots[0][1], 0))
        for i in range(1, len(slots)):
            if slots[i][1] < slots[i - 1][1]:
                r += 1
            ranked.append((slots[i][0], slots[i][1], r))
        agent_ranked_slots.append(ranked)
        max_rank_limit = max(max_rank_limit, r)

    def get_rank(agent, slot):
        for s, v, r in agent_ranked_slots[agent]:
            if s == slot:
                return r
        return 10**9

    def has_valid_upgrade(agent, src, tgt):
        return report_vals[agent, tgt[0], tgt[1]] > report_vals[agent, src[0], src[1]]

    def feasible_targets(agent, src, t_now):
        res = []
        src_r = get_rank(agent, src)
        for s, v, r in agent_ranked_slots[agent]:
            if r <= src_r and v > report_vals[agent, src[0], src[1]]:
                if t_now <= s[1] < cfg.T:
                    res.append(s)
        return res

    # ---- Promotion chain search ----
    def find_promotion_chain(t_now, real_cap):
        start_slots = []
        for ag in range(cfg.A):
            for m in range(cfg.M):
                for tt in range(t_now, cfg.T):
                    if allocation[ag, m, tt] > 0:
                        start_slots.append((ag, (m, tt)))

        rng.shuffle(start_slots)

        for ag0, src0 in start_slots:
            queue = deque()
            visited = set()
            queue.append((ag0, src0, []))
            visited.add((ag0, src0))

            while queue:
                ag, src, path = queue.popleft()

                for tgt in feasible_targets(ag, src, t_now):
                    tgt_m, tgt_t = tgt

                    # capacity check
                    if tgt_t == t_now:
                        cap_ok = allocation[:, :, tgt_t].sum() < real_cap
                    else:
                        cap_ok = allocation[:, :, tgt_t].sum() < max(
                            0, int(cfg.BASE_SUPPLY[tgt_t] - cfg.BUFFER[tgt_t])
                        )

                    mach_ok = allocation[:, tgt_m, tgt_t].sum() < cfg.MACHINE_CAPACITY

                    if cap_ok and mach_ok:
                        return path + [(ag, src, tgt)]

                    # relay: displace someone else
                    victims = []
                    if not mach_ok:
                        for v_ag in range(cfg.A):
                            if allocation[v_ag, tgt_m, tgt_t] > 0:
                                victims.append((v_ag, (tgt_m, tgt_t)))
                    elif not cap_ok:
                        for v_ag in range(cfg.A):
                            for v_m in range(cfg.M):
                                if allocation[v_ag, v_m, tgt_t] > 0:
                                    victims.append((v_ag, (v_m, tgt_t)))

                    for v_ag, v_src in victims:
                        if (v_ag, v_src) in visited:
                            continue
                        if has_valid_upgrade(v_ag, v_src, tgt):
                            visited.add((v_ag, v_src))
                            queue.append(
                                (v_ag, v_src, path + [(ag, src, tgt)])
                            )
        return None

    # ---- main loop ----
    for t in range(cfg.T):

        # Phase A: Window RSD（そのまま）
        window_end = min(cfg.T, t + cfg.WINDOW)
        order = list(range(cfg.A))
        rng.shuffle(order)

        for ag in order:
            while remaining_demand[ag] > 0:
                best = None
                best_val = 0
                for tt in range(t, window_end):
                    safe_cap = max(0, int(cfg.BASE_SUPPLY[tt] - cfg.BUFFER[tt]))
                    if allocation[:, :, tt].sum() < safe_cap:
                        for m in range(cfg.M):
                            if allocation[:, m, tt].sum() < cfg.MACHINE_CAPACITY:
                                v = report_vals[ag, m, tt]
                                if v > best_val:
                                    best_val = v
                                    best = (m, tt)
                if best is None:
                    break
                allocation[ag, best[0], best[1]] += 1
                remaining_demand[ag] -= 1

        # Phase B
        real_cap = min(
            cfg.M * cfg.MACHINE_CAPACITY,
            max(0, int(cfg.BASE_SUPPLY[t] + shocks[t]))
        )

        # ---- Case 1: Eviction（既存コードそのまま） ----
        if allocation[:, :, t].sum() > real_cap:
            # （あなたの既存コードをここにそのまま置く）
            pass

        # ---- Case 2: Promotion（全面刷新） ----
        while allocation[:, :, t].sum() < real_cap:
            chain = find_promotion_chain(t, real_cap)
            if chain is None:
                break

            # apply chain
            first_ag, first_src, _ = chain[0]
            allocation[first_ag, first_src[0], first_src[1]] -= 1
            for i, (ag, src, tgt) in enumerate(chain):
                allocation[ag, tgt[0], tgt[1]] += 1
                if i + 1 < len(chain):
                    next_ag, next_src, _ = chain[i + 1]
                    allocation[next_ag, next_src[0], next_src[1]] -= 1

        # ---- fallback: unassigned ----
        if allocation[:, :, t].sum() < real_cap:
            agents = list(range(cfg.A))
            rng.shuffle(agents)
            for ag in agents:
                if remaining_demand[ag] <= 0:
                    continue
                if allocation[:, :, t].sum() >= real_cap:
                    break
                best_m = None
                best_val = 0
                for m in range(cfg.M):
                    if allocation[:, m, t].sum() < cfg.MACHINE_CAPACITY:
                        v = report_vals[ag, m, t]
                        if v > best_val:
                            best_val = v
                            best_m = m
                if best_m is not None:
                    allocation[ag, best_m, t] += 1
                    remaining_demand[ag] -= 1

    return allocation
