import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Tuple

@dataclass(frozen=True)
class PairCorrelation:
    pair_id: Tuple[int, int]
    joint_correlations: dict[str, float]
    similarity_score: float =  field(init=False)

    def __post_init__(self) -> None:
        mean: float = float(np.mean(list(self.joint_correlations.values()))) if self.joint_correlations else 0.0
        object.__setattr__(self, 'similarity_score', mean)

    @classmethod
    def from_ids(cls, id_1: int, id_2: int, joint_correlations: dict[str, float]):
        pair_id: Tuple[int, int] = (id_1, id_2) if id_1 <= id_2 else (id_2, id_1)
        return cls(pair_id=pair_id, joint_correlations=joint_correlations)

@dataclass(frozen=True)
class PairCorrelationBatch:
    pair_correlations: List[PairCorrelation]
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    similarity: float = field(init=False)

    def __post_init__(self):
        mean: float = (
            sum(r.similarity_score for r in self.pair_correlations) / len(self.pair_correlations)
            if self.pair_correlations else 0.0
        )
        object.__setattr__(self, 'similarity', mean)

    @property
    def is_empty(self) -> bool:
        return len(self.pair_correlations) == 0

    @property
    def count(self) -> int:
        return len(self.pair_correlations)

    def get_most_similar_pair(self) -> Optional[PairCorrelation]:
        if not self.pair_correlations:
            return None
        return max(self.pair_correlations, key=lambda r: r.similarity_score)

PoseCorrelationBatchCallback = Callable[[PairCorrelationBatch], None]

