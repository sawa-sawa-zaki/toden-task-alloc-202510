from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Market:
    num_agents: int
    num_machines: int
    time_horizon: int
    units_demand: int

    supply_forecast_by_t: List[int]          # len=T
    buffer_by_t: List[int]                   # len=T
    machine_capacity: List[int]              # len=M
    supply_shock_by_t: List[Dict[str, int]]  # {t, delta}

    # 予測・バッファ
    def forecast_t(self, t: int) -> int:
        return self.supply_forecast_by_t[t]
    def buffer_t(self, t: int) -> int:
        return self.buffer_by_t[t] if self.buffer_by_t else 0
    def cap_t_for_rsd(self, t: int) -> int:
        return max(0, self.forecast_t(t) - self.buffer_t(t))

    # マシン別
    def sum_machine_cap(self) -> int:
        return sum(self.machine_capacity)
    def machine_cap(self, m: int) -> int:
        return self.machine_capacity[m]

    # 実供給（ショック適用後）の時刻総上限（電力×機械の双方でバインド）
    def total_cap_t_after(self, t: int) -> int:
        delta = sum(s["delta"] for s in self.supply_shock_by_t if s["t"] == t)
        return max(0, min(self.forecast_t(t) + delta, self.sum_machine_cap()))

    # RSDに渡す「時刻上限ベクトル」
    def cap_vec_for_rsd(self):
        return [self.cap_t_for_rsd(t) for t in range(self.time_horizon)]
