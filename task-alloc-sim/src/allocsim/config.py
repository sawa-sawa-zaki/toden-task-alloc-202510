from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import yaml
from copy import deepcopy

class MarketConfig(BaseModel):
    num_agents: Optional[int] = None
    num_machines: Optional[int] = None
    time_horizon: Optional[int] = None
    units_demand: int = 2

    # 事前予測とバッファ
    supply_forecast_by_t: Optional[List[int]] = None   # length = T
    buffer_by_t: Optional[List[int]] = None            # length = T（省略時0）

    # ★ マシン別キャパ（時間帯に依らず一定）
    machine_capacity: Optional[List[int]] = None       # length = M

    # 変動（±）
    supply_shock_by_t: List[Dict[str, int]] = Field(default_factory=list)  # {t, delta}

    # 互換用（未使用）
    supply_shock: List[Dict[str, int]] = Field(default_factory=list)

class PrefsConfig(BaseModel):
    type: str = "json_linear_values"
    path: Optional[str] = None
    value_dist: str = "normal"
    mean: float = 1.0
    std: float = 0.3
    allow_ties: bool = True
    tie_round: int = 2
    per_agent_correlation: float = 0.0
    seed: int = 0

class SimConfig(BaseModel):
    pipeline: List[str] = ["rsd_initial_global", "ttc_adjust_by_t"]
    time_horizon: int = 2
    repeats: int = 5
    seed: int = 42
    output_dir: str = "results/runs"
    log_interval: int = 10
    save_plots: bool = False

    market_defaults: MarketConfig
    market: MarketConfig
    prefs: PrefsConfig

def deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config_with_includes(path: str) -> dict:
    with open(path, "r") as f:
        root = yaml.safe_load(f)
    cfg: Dict[str, Any] = {}
    for inc in root.get("include", []):
        with open(inc, "r") as g:
            cfg = deep_merge(cfg, yaml.safe_load(g))
    cfg = deep_merge(cfg, {k: v for k, v in root.items() if k not in ("include", "overrides")})
    cfg = deep_merge(cfg, root.get("overrides", {}))
    return cfg
