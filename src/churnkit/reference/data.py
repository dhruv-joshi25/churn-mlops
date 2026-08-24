import pandas as pd

from churnkit.reference.schema import CATEGORICAL_FEATURES, FEATURES, ID_COL, TARGET


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning that must happen identically at train and serve time.

    Anything that learns from data (imputation values, scaling, encoding)
    belongs in the sklearn Pipeline, not here.
    """
    df = df.copy()

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # 11 rows have a blank string here; they are all tenure == 0.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def split_xy(df: pd.DataFrame):
    y = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    X = df[FEATURES]
    return X, y
