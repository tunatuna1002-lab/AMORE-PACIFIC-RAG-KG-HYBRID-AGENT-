"""Result fusion: one RRF, one weighted merge.

``rrf``      — reciprocal rank fusion (rank-space merging of ranked lists).
``weighted`` — per-item weighted scoring of the three HybridContext sources
               plus the aggregate ConfidenceFusion verdict.
"""

from .rrf import (
    DEFAULT_K,
    fuse,
    fuse_named,
    key_by_content_hash,
    key_by_id,
    key_by_id_or_rank,
)
from .weighted import (
    DEFAULT_RETRIEVAL_WEIGHTS,
    compute_fusion_confidence,
    load_retrieval_weights,
    weighted_merge,
)

__all__ = [
    "DEFAULT_K",
    "DEFAULT_RETRIEVAL_WEIGHTS",
    "compute_fusion_confidence",
    "fuse",
    "fuse_named",
    "key_by_content_hash",
    "key_by_id",
    "key_by_id_or_rank",
    "load_retrieval_weights",
    "weighted_merge",
]
