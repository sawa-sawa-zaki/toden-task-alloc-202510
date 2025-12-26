
from dataclasses import dataclass, field
from typing import List, Dict
import math

# 正規分布近似のショック分布（今回は使わないが互換性のため残す）
def generate_normal_shock(max_val: int) -> Dict[int, float]:
    probs = {}
    for k in range(-max_val, max_val + 1):
        probs[k] = math.exp(- (k ** 2) / (2 * max_val))
    Z = sum(probs.values())
    return {k: v / Z for k, v in probs.items()}


@dataclass
class Config:
    # --- 市場設定 ---
    A: int = 2              # エージェント数
    M: int = 2              # 計算機数
    T: int = 3              # 時間帯数
    WINDOW: int = 2         # タイムウィンドウ

    MACHINE_CAPACITY: int = 10

    # --- 戦略空間 ---
    STRATEGY_DOMAIN: str = "RESTRICTED"

    # --- 需要 ---
    Q: List[int] = field(default_factory=lambda: [20, 20])

    # --- 供給 ---
    BASE_SUPPLY: List[int] = field(default_factory=lambda: [10, 10, 10])
    BUFFER: List[float] = field(default_factory=lambda: [2, 2, 2])

    # --- ショック（今回は不使用）---
    SHOCK_PROB: Dict[int, float] = field(
        default_factory=lambda: generate_normal_shock(5)
    )


@dataclass
class ICConfig:
    # --- IC検証用 ---
    SAMPLES: int = 50
    SEED: int = 42
    MAX_PROFILES: int = None

    OUTPUT_BASE_DIR: str = "results"
