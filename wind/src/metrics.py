"""Metrics computed from stored prediction tables rather than during training.

summarise returns one row per run; per_class returns one row per class within a run;
frequency_recall_correlation reproduces the class-size versus recall relation.

Next to the classification scores, every run gets the scores of meteorological
forecast verification and of ordinal classification, which count class
distances in the scale order of the labels (Beaufort, or Q1..Q6): the Peirce
and Gerrity skill scores, quadratic-weighted kappa, errors in class steps and
the ranked probability score. The multi-category Heidke skill score is Cohen's
kappa, so it has no column of its own.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import class_labels

ID_COLUMNS = [
    "location",
    "model",
    "preprocessing",
    "features",
    "horizon",
    "splitter",
    "balancing",
    "classes",
    "source",
    "prior_correction",
]
DEFAULTS = {"balancing": "none", "classes": "beaufort", "prior_correction": "none"}
VERIFICATION = [
    "auc_ovo",
    "kappa_quadratic",
    "peirce",
    "gerrity",
    "mae_class",
    "amae",
    "mmae",
    "within1",
    "rps",
]


def load_predictions(directory: str | Path, pattern: str = "*") -> pd.DataFrame:
    """Every prediction file of a directory whose stem matches pattern, as one
    table with complete ID_COLUMNS.

    Files of default runs carry no balancing, classes, source or
    prior_correction column, so their rows get the defaults: none, beaufort,
    the location itself and none. Grouping by ID_COLUMNS, as
    tools/export_tables.py does, would otherwise drop them, because pandas
    skips NaN keys.
    """
    directory = Path(directory)
    files = sorted(directory.glob(f"{pattern}.parquet")) or sorted(directory.glob(f"{pattern}.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no prediction files in {directory}")
    frames = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files]
    frame = pd.concat(frames, ignore_index=True)
    for column, value in DEFAULTS.items():
        frame[column] = frame[column].fillna(value) if column in frame else value
    frame["source"] = frame["source"].fillna(frame["location"]) if "source" in frame else frame["location"]
    return frame


def _proba_columns(frame: pd.DataFrame) -> list[str]:
    """Probability columns of the rows at hand.

    A table that mixes class schemes leaves the other scheme's columns empty,
    so only complete columns count.
    """
    return [c for c in frame.columns if c.startswith("proba_") and frame[c].notna().all()]


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


def _scale(frame: pd.DataFrame) -> list[str]:
    """Labels of a run in scale order, classes it never met included."""
    labels = set(frame["y_true"].unique()) | set(frame["y_pred"].unique())
    return class_labels(labels | {c.removeprefix("proba_") for c in _proba_columns(frame)})


def _codes(values: pd.Series, labels: list[str]) -> np.ndarray:
    return pd.Categorical(values, categories=labels).codes.astype(np.intp)


def _mann_whitney(score: np.ndarray, positive: np.ndarray) -> float:
    """Binary AUC as the Mann-Whitney statistic, ties counted as half a win."""
    ranks = rankdata(score)
    n_positive = positive.sum()
    n_negative = len(positive) - n_positive
    return (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def _auc_ovo(frame: pd.DataFrame, labels: list[str], true: np.ndarray) -> float:
    """Hand and Till (2001) AUC: the mean over class pairs of both one-vs-one AUCs.

    A pair only compares hours of its own two classes, so unlike the
    one-vs-rest AUC the score does not move with the class priors of the test
    set. Pairs with a class absent from y_true are left out.
    """
    available = {c.removeprefix("proba_") for c in _proba_columns(frame)}
    present = [i for i, label in enumerate(labels) if label in available and (true == i).any()]
    if len(present) < 2:
        return float("nan")
    proba = {i: frame[f"proba_{labels[i]}"].to_numpy(dtype=float) for i in present}
    pairs = []
    for a, b in combinations(present, 2):
        mask = (true == a) | (true == b)
        is_a = true[mask] == a
        pairs.append((_mann_whitney(proba[a][mask], is_a) + _mann_whitney(proba[b][mask], ~is_a)) / 2)
    return float(np.mean(pairs))


def _kappa_quadratic(matrix: np.ndarray) -> float:
    """Cohen's kappa with weights growing with the squared class distance."""
    k = len(matrix)
    weights = np.subtract.outer(np.arange(k), np.arange(k)) ** 2
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / matrix.sum()
    chance = (weights * expected).sum()
    return float(1 - (weights * matrix).sum() / chance) if chance else float("nan")


def _peirce(matrix: np.ndarray) -> float:
    """Multi-category Peirce (Hanssen-Kuipers) skill score.

    Hits beyond chance, relative to the hits beyond chance of a perfect
    forecast; unlike kappa, the denominator depends on the observed class
    shares only.
    """
    joint = matrix / matrix.sum()
    observed, forecast = joint.sum(axis=1), joint.sum(axis=0)
    denominator = 1 - observed @ observed
    return float((np.trace(joint) - observed @ forecast) / denominator) if denominator else float("nan")


def _gerrity(matrix: np.ndarray) -> float:
    """Gerrity (1992) score, the equitable member of the Gandin-Murphy family.

    The scoring matrix comes from the observed class shares p_r through
    a_r = (1 - P_r) / P_r with P_r = p_1 + ... + p_r: a hit in a rare class is
    worth more, and a miss costs more the more classes it spans. A perfect
    forecast scores 1, a constant or a random one 0. An empty observed class
    leaves the matrix undefined, and the score is NaN.
    """
    k = len(matrix)
    observed = matrix.sum(axis=1) / matrix.sum()
    if k < 2 or (observed == 0).any():
        return float("nan")
    cumulative = np.cumsum(observed)[:-1]
    odds = (1 - cumulative) / cumulative
    scores = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            scores[i, j] = scores[j, i] = (1 / odds[:i]).sum() - (j - i) + odds[j:].sum()
    return float((matrix * scores).sum() / matrix.sum() / (k - 1))


def _ordinal(matrix: np.ndarray) -> dict:
    """Errors in class steps.

    mae_class averages over hours, amae over observed classes (the mean of the
    per-class MAE) and mmae is the worst class; within1 is the share of hours
    at most one class away from the truth.
    """
    k = len(matrix)
    distance = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    support = matrix.sum(axis=1)
    observed = support > 0
    per_class = (matrix * distance).sum(axis=1)[observed] / support[observed]
    return {
        "mae_class": float((matrix * distance).sum() / matrix.sum()),
        "amae": float(per_class.mean()),
        "mmae": float(per_class.max()),
        "within1": float(matrix[distance <= 1].sum() / matrix.sum()),
    }


def _rps(frame: pd.DataFrame, labels: list[str], true: np.ndarray) -> float:
    """Ranked probability score over the ordered classes, divided by K - 1 into [0, 1]."""
    if not _proba_columns(frame) or len(labels) < 2:
        return float("nan")
    proba = frame.reindex(columns=[f"proba_{label}" for label in labels]).fillna(0.0)
    forecast = np.cumsum(proba.to_numpy(dtype=float), axis=1)[:, :-1]
    observed = true[:, None] <= np.arange(len(labels) - 1)
    return float(((forecast - observed) ** 2).sum(axis=1).mean() / (len(labels) - 1))


def _row(frame: pd.DataFrame) -> dict:
    y_true, y_pred = frame["y_true"], frame["y_pred"]
    counts = y_true.value_counts()
    labels = _scale(frame)
    true, predicted = _codes(y_true, labels), _codes(y_pred, labels)
    k = len(labels)
    matrix = np.bincount(true * k + predicted, minlength=k * k).reshape(k, k)
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
        "auc_ovo": _auc_ovo(frame, labels, true),
        "kappa_quadratic": _kappa_quadratic(matrix),
        "peirce": _peirce(matrix),
        "gerrity": _gerrity(matrix),
        **_ordinal(matrix),
        "rps": _rps(frame, labels, true),
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

    out = overall.drop(columns=VERIFICATION).join(
        fold_wise[["accuracy_fold_std", "kappa_fold_std", "macro_f1_fold_std"]]
    )
    out["lift_over_zero_r"] = out["accuracy"] - out["zero_r"]
    out = out.join(overall[VERIFICATION])
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
            predicted = matrix[:, i].sum()
            fp = predicted - tp
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
                    "predicted": int(predicted),
                    "frequency_bias": predicted / support[i] if support[i] else float("nan"),
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
