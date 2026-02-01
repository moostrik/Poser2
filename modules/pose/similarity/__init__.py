# Import features first to avoid circular imports
from .features.SimilarityBatch import SimilarityBatch, SimilarityBatchCallback
from .features.SimilarityFeature import SimilarityFeature, AggregationMethod
from .features.SimilarityStream import SimilarityStream

# Import similarity computers after features are available
from .FrameSimilarity import FrameSimilarity
from .WindowSimilarity import WindowSimilarity, WindowSimilarityConfig