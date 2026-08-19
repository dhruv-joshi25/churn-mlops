from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import CATEGORICAL_FEATURES, CATEGORY_VALUES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Fit this INSIDE cross-validation folds, never on the full dataset.

    Categories are pinned from schema.py rather than inferred, so an unseen
    value at serve time cannot silently shift column positions.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    categories=[CATEGORY_VALUES[c] for c in CATEGORICAL_FEATURES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    return list(fitted_preprocessor.get_feature_names_out())
