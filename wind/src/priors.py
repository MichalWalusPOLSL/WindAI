"""Prior-shift corrections of stored class probabilities.

A probabilistic classifier estimates p(class | x) under the class priors of its
training set. Under other priors, Bayes' rule rescales each probability by the
ratio of the new prior to the old one and renormalises (Saerens, Latinne and
Decaestecker 2002). The part of an error that the correction removes is owed
to the class frequencies; what it leaves is owed to classes that overlap in
feature space. apply takes the new priors from one of three places:

- uniform: equal priors, which takes the training priors out of the decision;
- oracle: the true class shares of the test fold, the best any prior
  correction can do;
- em: estimated from the unlabelled probabilities alone.

The correction assumes calibrated probabilities, so use it on xgb or gbt.
Random forest probabilities are vote shares pulled towards the middle, and
rescaling them answers a different question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import ID_COLUMNS

EPS = 1e-6
METHODS = ("none", "uniform", "oracle", "em")


def adjust(proba, train_prior, target_prior) -> np.ndarray:
    """Rescale p_k by t_k / s_k and renormalise each row, priors floored at EPS."""
    ratio = np.maximum(np.asarray(target_prior, dtype=float), EPS) / np.maximum(
        np.asarray(train_prior, dtype=float), EPS
    )
    scaled = np.asarray(proba, dtype=float) * ratio
    return scaled / scaled.sum(axis=1, keepdims=True)


def em(proba, train_prior, tol: float = 1e-6, max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Test priors estimated by EM from unlabelled probabilities (Saerens et al. 2002).

    Starting from the training priors, the E step adjusts the probabilities to
    the current estimate and the M step replaces the estimate with their mean,
    until no prior moves by more than tol. Returns the estimated priors and the
    probabilities adjusted to them.
    """
    prior = np.asarray(train_prior, dtype=float)
    for _ in range(max_iter):
        previous, prior = prior, adjust(proba, train_prior, prior).mean(axis=0)
        if np.abs(prior - previous).max() < tol:
            break
    return prior, adjust(proba, train_prior, prior)


def _train_prior(run: pd.DataFrame, rows: pd.DataFrame, fold, labels: list[str]) -> np.ndarray:
    """Class priors of the training set behind one fold of a run.

    Files written before run.py stored them get them back from the labels:
    under LOYO the training set of a fold is exactly the other folds.
    """
    columns = [f"prior_{label}" for label in labels]
    if all(c in rows and rows[c].notna().all() for c in columns):
        return rows[columns].iloc[0].to_numpy(dtype=float)
    if (run["splitter"] != "loyo").any():
        raise ValueError(
            "a run without prior_ columns can only be corrected under LOYO, where the "
            "training set of a fold is the other folds"
        )
    others = run.loc[run["fold"] != fold, "y_true"]
    return others.value_counts(normalize=True).reindex(labels, fill_value=0.0).to_numpy(dtype=float)


def apply(predictions: pd.DataFrame, method: str) -> pd.DataFrame:
    """Prior-corrected copy of a prediction table, corrected per run and per fold.

    The corrected probabilities replace the proba_ columns, y_pred becomes
    their argmax and prior_correction records the method. With "none" only the
    column is added, so the uncorrected table can be stacked with the
    corrected ones and summarised in one go.
    """
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; available: {list(METHODS)}")
    out = predictions.copy()
    out["prior_correction"] = method
    if method == "none":
        return out
    proba_columns = [c for c in out.columns if c.startswith("proba_")]
    out[proba_columns] = out[proba_columns].astype(float)

    run_keys = [c for c in ID_COLUMNS if c in out and c != "prior_correction"]
    for _, run in out.groupby(run_keys, observed=True, dropna=False):
        columns = [c for c in run.columns if c.startswith("proba_") and run[c].notna().all()]
        if not columns:
            raise ValueError("a prior correction needs the class probabilities of the run")
        labels = [c.removeprefix("proba_") for c in columns]
        for fold, rows in run.groupby("fold"):
            proba = rows[columns].to_numpy(dtype=float)
            train_prior = _train_prior(run, rows, fold, labels)
            if method == "em":
                _, corrected = em(proba, train_prior)
            elif method == "uniform":
                corrected = adjust(proba, train_prior, np.full(len(labels), 1 / len(labels)))
            else:
                truth = rows["y_true"].value_counts(normalize=True).reindex(labels, fill_value=0.0)
                corrected = adjust(proba, train_prior, truth.to_numpy(dtype=float))
            out.loc[rows.index, columns] = corrected
            out.loc[rows.index, "y_pred"] = np.asarray(labels, dtype=object)[corrected.argmax(axis=1)]
    return out
