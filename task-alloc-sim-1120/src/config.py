from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Config:
    # --- 市場設定 ---
    A: int = 2              # エージェント数
    M: int = 2              # 計算機数
    T: int = 3              # 時間帯数
    WINDOW: int = 2         # タイムウィンドウの大きさ
    
    # ★重要: 計算機1台あたりの物理容量 (Supply=10〜20の実験に耐えるよう大きく設定)
    MACHINE_CAPACITY: int = 20

    # --- 需要設定 ---
    # 各自が実行したいタスクの総量
    Q: List[int] = field(default_factory=lambda: [20, 20])
    
    # --- 供給・バッファ設定 ---
    # BASE_SUPPLY: 予測される供給量
    BASE_SUPPLY: List[int] = field(default_factory=lambda: [10, 10, 10]) 
    
    # BUFFER: RSDで配分せずに取っておく量 (実供給が下振れした時の保険)
    # 例: [0.2, 0.2, 0.2] とすると、RSDでは0.8までしか配分しない
    BUFFER: List[float] = field(default_factory=lambda: [5, 5, 5]) 
    
    # --- ショック確率 ---
    # キー: 変動量, 値: 確率
    SHOCK_PROB: Dict[int, float] = field(default_factory=lambda: {
        -5: 0.25, 
        0:  0.55, 
        5:  0.25
    })

@dataclass
class ICConfig:
    SAMPLES: int = 50       # Outcome Matrix構築時の試行回数
    SEED: int = 42          # 共通乱数シード
    MAX_PROFILES: int = None # 検証する真のプロファイル数 (Noneで全探索)
    
    # 結果出力の親ディレクトリ (この下に日時フォルダが作られます)
    OUTPUT_BASE_DIR: str = "results"