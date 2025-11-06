import numpy as np


def generate_linear_values(
    num_agents: int,
    num_machines: int,
    time_horizon: int,
    dist: str = "normal",
    mean: float = 1.0,
    std: float = 0.3,
    allow_ties: bool = True,
    tie_round: int = 2,
    per_agent_correlation: float = 0.0,
    seed: int = 0,
):
    """
    戻り値: values[agent, machine, t] の実数行列。
    allow_ties=True の場合は丸めによって無差別を発生させる。
    """
    rng = np.random.default_rng(seed)
    vals = np.zeros((num_agents, num_machines, time_horizon), dtype=float)

    if dist == "normal":
        base = rng.normal(mean, std, size=(num_agents, num_machines, time_horizon))
    else:
        base = rng.uniform(mean - std, mean + std, size=(num_agents, num_machines, time_horizon))

    if per_agent_correlation > 0:
        # 各agentに共通ショック + 独自ショック
        common = rng.normal(0, std, size=(num_agents, 1, 1))
        base = (per_agent_correlation * common) + ((1 - per_agent_correlation) * base)

    vals = base
    if allow_ties and tie_round is not None:
        vals = np.round(vals, tie_round)

    return vals
