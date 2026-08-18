"""Metrics computed from stored prediction tables rather than during training.

summarise returns one row per run; per_class returns one row per class within a run;
frequency_recall_correlation reproduces the class-size versus recall relation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ID_COLUMNS = ["location", "model", "preprocessing", "horizon", "splitter"]


def load_predictions(directory: str | Path) -> pd.DataFrame:
    directory = Path(directory)
    files = sorted(directory.glob("*.parquet")) or sorted(directory.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no prediction files in {directory}")
    frames = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def _proba_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("proba_")]


def _auc(frame: pd.DataFrame) -> float:
    columns = _proba_columns(frame)
    if not columns:
        return float("nan")
    labels = [c.removeprefix("proba_") for c in columns]
    present = frame["y_true"].unique()
    if len(present) < 2:
        return float("nan")
    try:
        return roc_auc_score(
            frame["y_true"], frame[columns].to_numpy(), multi_class="ovr",
            average="macro", labels=labels,
        )
    except ValueError:
        return float("nan")


def _row(frame: pd.DataFrame) -> dict:
    y_true, y_pred = frame["y_true"], frame["y_pred"]
    counts = y_true.value_counts()
    return {
        "n": len(frame),
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_ovr": _auc(frame),
        "zero_r": counts.max() / len(frame),
    }


def summarise(predictions: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = by or ID_COLUMNS
    by = [c for c in by if c in predictions.columns]

    overall = predictions.groupby(by, observed=True).apply(
        lambda g: pd.Series(_row(g)), include_groups=False
    )

    fold_wise = (
        predictions.groupby(by + ["fold"], observed=True)
        .apply(lambda g: pd.Series(_row(g)), include_groups=False)
        .groupby(by, observed=True)
        .agg(["mean", "std"])
    )
    fold_wise.columns = [f"{a}_fold_{b}" for a, b in fold_wise.columns]

    out = overall.join(fold_wise[["accuracy_fold_std", "kappa_fold_std", "macro_f1_fold_std"]])
    out["lift_over_zero_r"] = out["accuracy"] - out["zero_r"]
    return out.reset_index().sort_values("kappa", ascending=False)


def per_class(predictions: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = [c for c in (by or ID_COLUMNS) if c in predictions.columns]
    rows = []
    for keys, group in predictions.groupby(by, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        labels = sorted(set(group["y_true"]) | set(group["y_pred"]))
        matrix = confusion_matrix(group["y_true"], group["y_pred"], labels=labels)
        support = matrix.sum(axis=1)
        for i, label in enumerate(labels):
            tp = matrix[i, i]
            fp = matrix[:, i].sum() - tp
            fn = support[i] - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / support[i] if support[i] else 0.0
            rows.append(
                dict(zip(by, keys))
                | {
                    "label": label,
                    "support": int(support[i]),
                    "share": support[i] / support.sum(),
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                }
            )
    return pd.DataFrame(rows)


def confusion(predictions: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or sorted(set(predictions["y_true"]) | set(predictions["y_pred"]))
    matrix = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def frequency_recall_correlation(class_table: pd.DataFrame) -> dict:
    share, recall = class_table["share"], class_table["recall"]
    return {
        "n_points": len(class_table),
        "pearson": float(np.corrcoef(share, recall)[0, 1]),
        "spearman": float(share.corr(recall, method="spearman")),
    }
