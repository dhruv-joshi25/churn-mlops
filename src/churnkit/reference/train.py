"""Training entrypoint with MLflow tracking.

Run:  python -m churnkit.reference.train --model xgboost
The point of this file is that every run is logged identically, so runs are
comparable in the MLflow UI. Sanjida's notebook experiments can call
train_and_log() directly.
"""

import argparse

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from churnkit import config
from churnkit.reference.data import clean, load_raw, split_xy
from churnkit.reference.preprocess import build_preprocessor

# Business costs used to pick the decision threshold. Change these with the
# team; they matter more than any hyperparameter.
COST_RETENTION_OFFER = 200.0   # cost of giving a discount to one customer
VALUE_CUSTOMER_SAVED = 1200.0  # value of retaining a customer who would churn


def build_estimator(name: str, scale_pos_weight: float):
    if name == "logistic":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
        )
    raise ValueError(f"unknown model: {name}")


def pick_threshold(y_true, y_prob) -> tuple[float, float]:
    """Choose the cutoff that maximises expected profit, not F1."""
    best_t, best_value = 0.5, -np.inf
    for t in np.linspace(0.05, 0.95, 91):
        flagged = y_prob >= t
        tp = int(((flagged) & (y_true == 1)).sum())
        fp = int(((flagged) & (y_true == 0)).sum())
        value = tp * VALUE_CUSTOMER_SAVED - (tp + fp) * COST_RETENTION_OFFER
        if value > best_value:
            # Round here: linspace yields values like 0.36999999999999994, which
            # then surface in API responses and the report.
            best_t, best_value = round(float(t), 4), float(value)
    return best_t, best_value


def train_and_log(model_name: str = "xgboost", register: bool = False):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df = clean(load_raw(config.DATA_PATH))
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", build_estimator(model_name, pos_weight)),
        ]
    )

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(
            {
                "model": model_name,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "churn_rate_train": round(float(y_train.mean()), 4),
                "scale_pos_weight": round(pos_weight, 3),
                "cost_offer": COST_RETENTION_OFFER,
                "value_saved": VALUE_CUSTOMER_SAVED,
            }
        )

        # Threshold is chosen on out-of-fold train predictions, never on test.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = cross_val_predict(
            pipe, X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        threshold, expected_value = pick_threshold(y_train.to_numpy(), oof)

        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= threshold).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, prob),
            "pr_auc": average_precision_score(y_test, prob),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "threshold": threshold,
            "oof_expected_value": expected_value,
        }
        mlflow.log_metrics(metrics)
        mlflow.set_tag("decision_threshold", threshold)

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=X_train.head(3),
            registered_model_name=config.REGISTERED_MODEL_NAME if register else None,
        )

        print(f"run_id={run.info.run_id}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return run.info.run_id, metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="xgboost",
                   choices=["logistic", "random_forest", "xgboost"])
    p.add_argument("--register", action="store_true")
    args = p.parse_args()
    train_and_log(args.model, args.register)
