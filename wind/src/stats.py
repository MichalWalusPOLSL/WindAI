"""Paired significance tests over stored predictions.

The fold-level tests treat each test year as one observation: fold_scores
builds the table of per-fold scores, and sign_test, wilcoxon, bootstrap_years
and friedman_nemenyi compare models on it. Conclusions rest on these. mcnemar
compares two models hour by hour within one location and is kept as an
auxiliary check only: at a lag-1 autocorrelation of 0.98 the hours are far
from independent, and its p-values come out much too small.
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
    """Exact McNemar test on the hours exactly one of the two models gets right.

    The test assumes independent hours. Hourly weather is persistent, so the
    effective sample is a small fraction of the hour count and the p-values
    are far too optimistic: use it as an auxiliary check next to the
    fold-level tests, never as the evidence.
    """
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


def fold_scores(
    summary_per_fold: pd.DataFrame, metric: str = "kappa", index: str | list[str] = "fold"
) -> pd.DataFrame:
    """Per-block scores, one row per block and one column per model.

    A block is a test year with index="fold", or a location-year with
    index=["location", "fold"]: 27 blocks for three locations and nine years.
    Several rows in one block and model are refused instead of averaged, which
    is what happened silently when several locations met index="fold".
    """
    keys = [index] if isinstance(index, str) else list(index)
    if summary_per_fold.duplicated(keys + ["model"]).any():
        raise ValueError(
            f"several rows share {keys + ['model']}; filter the table to one run per "
            "model or add the varying column to index, e.g. index=['location', 'fold']"
        )
    return summary_per_fold.pivot_table(index=index, columns="model", values=metric)


def sign_test(scores: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """Two-sided binomial test on the number of blocks each model wins.

    Each block counts once whatever the margin, so nothing is assumed about
    the distribution of the differences. Ties are dropped.
    """
    difference = (scores[model_a] - scores[model_b]).dropna()
    wins_a, wins_b = int((difference > 0).sum()), int((difference < 0).sum())
    p_value = stats.binomtest(wins_a, wins_a + wins_b, 0.5).pvalue if wins_a + wins_b else 1.0
    return {
        "model_a": model_a,
        "model_b": model_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": int((difference == 0).sum()),
        "p_value": float(p_value),
    }


def bootstrap_years(
    scores: pd.DataFrame, model_a: str, model_b: str, n: int = 10000, seed: int = 1992
) -> dict:
    """Bootstrap of the mean score difference a - b over resampled test years.

    Folds are drawn with replacement. With a (location, fold) index every
    location of a drawn year comes along: the three locations share the
    weather of a year and are not independent blocks. Returns the observed
    mean difference, its 95% percentile interval and the share of resamples
    in which a beats b.
    """
    difference = (scores[model_a] - scores[model_b]).dropna()
    years = difference.index.get_level_values("fold") if "fold" in difference.index.names else difference.index
    per_year = difference.groupby(np.asarray(years)).agg(["sum", "size"])
    draws = np.random.default_rng(seed).integers(0, len(per_year), size=(n, len(per_year)))
    means = per_year["sum"].to_numpy()[draws].sum(axis=1) / per_year["size"].to_numpy()[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_years": len(per_year),
        "mean_difference": float(difference.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_greater": float((means > 0).mean()),
    }


def wilcoxon(scores: pd.DataFrame, model_a: str, model_b: str) -> dict:
    statistic, p_value = stats.wilcoxon(scores[model_a], scores[model_b])
    return {
        "model_a": model_a,
        "model_b": model_b,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "median_difference": float((scores[model_a] - scores[model_b]).median()),
    }
