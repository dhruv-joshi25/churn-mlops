"""Features computed strictly backwards from a point in time.

Platform code: no column name, category or business rule from any particular
dataset may appear here (I11).
"""

from churnkit.features.events import (
    FEATURE_GROUPS,
    FeatureMatrix,
    build_features,
)

__all__ = ["FEATURE_GROUPS", "FeatureMatrix", "build_features"]
