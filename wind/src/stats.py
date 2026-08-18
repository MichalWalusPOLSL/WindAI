"""Paired significance tests over stored predictions.

mcnemar compares two models within one location; friedman_nemenyi ranks several
models across folds. Both expect the prediction table produced by run.py.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def _aligned(predictions: pd.DataFrame, model_a: str, model_b: str, **filters):
    frame = predictions
    for key, value in filters.items():
        frame = frame[frame[key] == value]
    a = frame[frame["model"] == model_a].sort_values("timestamp")
    b = frame[frame["model"] == model_b].sort_values("timestamp")
    if len(a) != len(b) or not np.array_equal(a["timestamp"].to_numpy(), b["timestamp"].to_numpy()):
        raise ValueError(f"predictions for {model_a} and {model_b} are not aligned")
    return a, b


def mcnemar(predictions: pd.DataFrame, model_a: str, model_b: str, **filters) -> dict:
    a, b = _aligned(predictions, model_a, model_b, **filters)
    correct_a = (a["y_pred"].to_numpy() == a["y_true"].to_numpy())
    correct_b = (b["y_pred"].to_numpy() == b["y_true"].to_numpy())

    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))

    if n01 + n10 == 0:
        return {"model_a": model_a, "model_b": model_b, "n01": 0, "n10": 0,
                "statistic": 0.0, "p_value": 1.0}

    result = stats.binomtest(n01, n01 + n10, 0.5)
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "n01": n01,
        "n10": n10,
        "statistic": float(statistic),
        "p_value": float(result.pvalue),
    }


def mcnemar_matrix(predictions: pd.DataFrame, models: list[str] | None = None, **filters) -> pd.DataFrame:
    models = models or sorted(predictions["model"].unique())
    rows = [mcnemar(predictions, a, b, **filters) for a, b in combinations(models, 2)]
    frame = pd.DataFrame(rows)
    frame["p_holm"] = _holm(frame["p_value"].to_numpy())
    return frame


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def friedman_nemenyi(scores: pd.DataFrame, alpha: float = 0.05) -> dict:
    matrix = scores.to_numpy()
    if matrix.shape[1] < 3:
        raise ValueError(
            f"Friedman needs at least three models, got {matrix.shape[1]}; "
            "use wilcoxon() for a pair"
        )
    if matrix.shape[0] < 3:
        raise ValueError("Friedman needs at least three folds")

    statistic, p_value = stats.friedmanchisquare(*matrix.T)
    ranks = np.apply_along_axis(stats.rankdata, 1, -matrix)
    mean_ranks = pd.Series(ranks.mean(axis=0), index=scores.columns).sort_values()

    n, k = matrix.shape
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}.get(k)
    critical = q_alpha * np.sqrt(k * (k + 1) / (6 * n)) if q_alpha else float("nan")

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_ranks": mean_ranks,
        "critical_difference": float(critical),
        "alpha": alpha,
    }


def fold_scores(summary_per_fold: pd.DataFrame, metric: str = "kappa") -> pd.DataFrame:
    return summary_per_fold.pivot_table(index="fold", columns="model", values=metric)


def wilcoxon(scores: pd.DataFrame, model_a: str, model_b: str) -> dict:
    statistic, p_value = stats.wilcoxon(scores[model_a], scores[model_b])
    return {
        "model_a": model_a,
        "model_b": model_b,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "median_difference": float((scores[model_a] - scores[model_b]).median()),
    }
