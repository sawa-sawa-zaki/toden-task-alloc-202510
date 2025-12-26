
from datetime import datetime
import os

def make_timestamped_dir(base_dir: str, prefix: str):
    """
    実験結果を必ずタイムスタンプ付きディレクトリに保存するためのヘルパー。
    例: results/expA_20250101_235959/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"{prefix}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path
