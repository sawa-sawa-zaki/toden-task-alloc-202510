from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

# --- ショック分布生成用のヘルパー関数 ---

def generate_uniform_shock(magnitude: int) -> Dict[int, float]:
    """
    一様分布: -magnitude から +magnitude までの整数すべてに等しい確率を割り振る
    例: magnitude=2 -> {-2: 0.2, -1: 0.2, 0: 0.2, 1: 0.2, 2: 0.2}
    """
    if magnitude == 0:
        return {0: 1.0}
        
    count = (magnitude * 2) + 1
    prob = 1.0 / count
    return {i: prob for i in range(-magnitude, magnitude + 1)}

def generate_normal_shock(magnitude: int, sigma_scale: float = 2.5) -> Dict[int, float]:
    """
    正規分布(近似): 0を中心に山なりの確率を割り振る
    magnitude: 最大振れ幅 (これを超える確率は切り捨てて正規化)
    sigma_scale: 標準偏差の調整 (magnitude / sigma_scale = sigma)
                 値が大きいほど鋭い山(0に集中)、小さいほどなだらかになる
    """
    if magnitude == 0:
        return {0: 1.0}

    probs = {}
    total_weight = 0.0
    
    # 標準偏差の設定 (最大値の約半分を1シグマとする設定)
    sigma = max(1.0, magnitude / sigma_scale)
    
    for i in range(-magnitude, magnitude + 1):
        # 確率密度関数 (PDF) の計算
        weight = np.exp(- (i**2) / (2 * sigma**2))
        probs[i] = weight
        total_weight += weight
        
    # 合計が1になるように正規化
    for k in probs:
        probs[k] /= total_weight
        
    return probs

# --- 設定クラス ---

@dataclass
class Config:
    # --- 市場設定 ---
    A: int = 2              # エージェント数
    M: int = 2              # 計算機数
    T: int = 3              # 時間帯数
    WINDOW: int = 2         # タイムウィンドウ
    
    # 計算機1台あたりの物理容量
    # (Supply=10〜20の実験を行う場合、これがボトルネックにならないよう十分大きくする)
    MACHINE_CAPACITY: int = 10
    
    # --- 戦略空間の設定 ---
    # "RESTRICTED":         一貫性のある選好(辞退含む)のみ
    # "UNRESTRICTED":       厳密な順序のみ(720通り) ※推奨
    # "UNRESTRICTED_WEAK":  無差別を含む全順序(4.6万通り) ※激重
    STRATEGY_DOMAIN: str = "RESTRICTED"
    
    # --- 需要設定 ---
    # 各自のタスク総量
    Q: List[int] = field(default_factory=lambda: [20, 20])
    
    # --- 供給・バッファ ---
    # BASE_SUPPLY: 予測される供給電力
    BASE_SUPPLY: List[int] = field(default_factory=lambda: [10, 10, 10]) 
    # BUFFER: 安全マージン
    BUFFER: List[float] = field(default_factory=lambda: [2, 2, 2]) 
    
    # --- ショック確率分布 ---
    # デフォルトで「最大値5の正規分布」を使うように設定
    # (main.py や debug_run.py はこのデフォルト値を使用する)
    SHOCK_PROB: Dict[int, float] = field(
        default_factory=lambda: generate_normal_shock(5)
    )

@dataclass
class ICConfig:
    # --- IC検証用設定 ---
    SAMPLES: int = 50       # シミュレーション試行回数 (モンテカルロ用)
    SEED: int = 42          # 乱数シード
    MAX_PROFILES: int = None # 検証する真のプロファイル数 (Noneなら全探索)
    
    # 結果出力の親ディレクトリ
    OUTPUT_BASE_DIR: str = "results"