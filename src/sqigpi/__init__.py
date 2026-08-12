"""Core utilities for scale-quotient information geometry."""
from .core import (
    fisher_blocks,
    horizontal_lift,
    quotient_metric,
    authority_matrix,
    eigendirectional_authority,
)

__all__ = [
    "fisher_blocks",
    "horizontal_lift",
    "quotient_metric",
    "authority_matrix",
    "eigendirectional_authority",
]
__version__ = "1.0.0"
