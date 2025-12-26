# main_exp_cx.py
"""
Type Cx（厳密順序 720）専用実験のエントリポイント。

- 真の選好：RESTRICTED（216）
- 嘘：Type Cx（厳密順序 720）
- A 側のみが嘘をつくケースを検証
"""

from src.experiments.exp_a_strategy_scan_typeCx import run

if __name__ == "__main__":
    run()
