
import random

def make_rng(seed: int):
    """乱数生成器を一元管理するためのヘルパー"""
    rng = random.Random(seed)
    return rng
