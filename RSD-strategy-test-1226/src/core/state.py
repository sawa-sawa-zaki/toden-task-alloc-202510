
# 割当状態と残需要を管理するクラス

from dataclasses import dataclass
from typing import List

@dataclass
class State:
    # allocation[t][i] = 1 if agent i is assigned at time t
    allocation: List[List[int]]
    remaining: List[int]

    @classmethod
    def init(cls, A: int, T: int, Q: List[int]):
        allocation = [[0 for _ in range(A)] for _ in range(T)]
        return cls(allocation=allocation, remaining=Q.copy())
