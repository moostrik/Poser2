import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Tuple

@dataclass(frozen=True)
class PairCorrelation:
    pair_id: Tuple[int, int]
    correlations: dict[str, float]
    mean_correlation: float =  field(init=False)

    def __post_init__(self) -> None:
        mean: float = float(np.mean(list(self.correlations.values()))) if self.correlations else 0.0
        object.__setattr__(self, 'mean_correlation', mean)

    @classmethod
    def from_ids(cls, id_1: int, id_2: int, correlations: dict[str, float]):
        pair_id: Tuple[int, int] = (id_1, id_2) if id_1 <= id_2 else (id_2, id_1)
        return cls(pair_id=pair_id, correlations=correlations)

@dataclass(frozen=True)
class PairCorrelationBatch:
    pair_correlations: List[PairCorrelation]
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    mean_correlation: float = field(init=False)

    def __post_init__(self):
        mean: float = (
            sum(r.mean_correlation for r in self.pair_correlations) / len(self.pair_correlations)
            if self.pair_correlations else 0.0
        )
        object.__setattr__(self, 'mean_correlation', mean)

    @property
    def is_empty(self) -> bool:
        return len(self.pair_correlations) == 0

    @property
    def count(self) -> int:
        return len(self.pair_correlations)

    def get_most_correlated_pair(self) -> Optional[PairCorrelation]:
        if not self.pair_correlations:
            return None
        return max(self.pair_correlations, key=lambda r: r.mean_correlation)

    def get_mean_correlation_for_pair(self, pair_id: tuple[int, int]) -> float:
        id1, id2 = pair_id
        pair_id = (id1, id2) if id1 <= id2 else (id2, id1)
        for pc in self.pair_correlations:
            if pc.pair_id == pair_id:
                return pc.mean_correlation
        return 0.0

PoseCorrelationBatchCallback = Callable[[PairCorrelationBatch], None]
