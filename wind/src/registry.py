"""Registry of preprocessing variants and model factories.

Models are lazy factories so that optional GPU backends (xgboost, lightgbm) do
not need to be installed for the rest of the registry to work. Their device is
probed at build time, so the same config runs on a CUDA box and on a CPU-only
laptop. Ensembles receive the outer splitter so that inner out-of-fold
predictions stay time-aware.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC

SEED = 1992


def _columns(X):
    categorical = [c for c in X.columns if str(X[c].dtype) == "category"]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def make_preprocessor(variant: str, X) -> ColumnTransformer | str:
    numeric, categorical = _columns(X)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if variant == "raw":
        numeric_step = "passthrough"
    elif variant == "scaled":
        numeric_step = StandardScaler()
    elif variant == "binned_width":
        numeric_step = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")
    elif variant == "binned_freq":
        numeric_step = KBinsDiscretizer(
            n_bins=10, encode="ordinal", strategy="quantile", subsample=None
        )
    else:
        raise KeyError(f"unknown preprocessing variant {variant!r}")

    return ColumnTransformer(
        [("num", numeric_step, numeric), ("cat", encoder, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


PREPROCESSING = ["raw", "scaled", "binned_width", "binned_freq"]


def _random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        criterion="gini",
        n_jobs=-1,
        random_state=SEED,
    )


def _hist_gbt():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=None, random_state=SEED
    )


def _svm():
    return SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED)


def _knn():
    return KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)


def _logreg():
    return LogisticRegression(max_iter=2000, random_state=SEED)


class LabelEncoded(ClassifierMixin, BaseEstimator):
    """Adapt an estimator that only accepts integer targets to string labels.

    XGBoost refuses the Beaufort strings ("Bf0-1", ...) that every other model,
    metric and prediction file uses, so the encoding is hidden here instead of
    leaking a second label convention into the stored predictions.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def _gpu_device(library: str, requested: str, fallback: str) -> str:
    """Return `requested` when that library really can train on the GPU here.

    The PyPI wheels differ: xgboost ships CUDA kernels, lightgbm does not, and a
    machine without an NVIDIA card has neither. WIND_XGB_DEVICE / WIND_LGBM_DEVICE
    override the probe when a run must be pinned to one backend.
    """
    override = os.environ.get(f"WIND_{library.upper()}_DEVICE")
    if override:
        return override
    return requested if _gpu_works(library, requested) else fallback


@contextmanager
def _muted_stderr():
    """Silence the native stderr of a failing probe, which prints before raising."""
    saved = os.dup(2)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)


@lru_cache(maxsize=None)
def _gpu_works(library: str, device: str) -> bool:
    """Fit a throwaway model on `device`; the probe costs less than one fold."""
    import numpy as np

    X = np.zeros((4, 2), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    try:
        with _muted_stderr():
            if library == "xgb":
                from xgboost import XGBClassifier

                XGBClassifier(n_estimators=1, device=device, verbosity=0).fit(X, y)
            else:
                from lightgbm import LGBMClassifier

                LGBMClassifier(
                    n_estimators=1, device=device, verbose=-1, min_child_samples=1
                ).fit(X, y)
    except Exception:
        return False
    return True


def _xgboost():
    from xgboost import XGBClassifier

    return LabelEncoded(
        XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.1,
            tree_method="hist",
            device=_gpu_device("xgb", "cuda", "cpu"),
            random_state=SEED,
        )
    )


def _lightgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.1,
        device=_gpu_device("lgbm", "gpu", "cpu"),
        random_state=SEED,
    )


BASE_MODELS: dict[str, Callable] = {
    "rf": _random_forest,
    "gbt": _hist_gbt,
    "svm": _svm,
    "knn": _knn,
    "logreg": _logreg,
    "nb": GaussianNB,
    "lda": LinearDiscriminantAnalysis,
    "xgb": _xgboost,
    "lgbm": _lightgbm,
}

ENSEMBLE_MEMBERS = ["rf", "gbt", "knn"]

SCALE_SENSITIVE = frozenset({"svm", "knn", "logreg", "lda"})


def _voting(cv=None):
    return VotingClassifier(
        [(name, BASE_MODELS[name]()) for name in ENSEMBLE_MEMBERS], voting="soft", n_jobs=1
    )


def _stacking(cv=None):
    if cv is None:
        raise ValueError("stacking requires an explicit time-aware cv splitter")
    return StackingClassifier(
        estimators=[(name, BASE_MODELS[name]()) for name in ENSEMBLE_MEMBERS],
        final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
        cv=cv,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )


ENSEMBLES: dict[str, Callable] = {"voting": _voting, "stacking": _stacking}


def build_model(name: str, X, preprocessing: str = "raw", cv=None) -> Pipeline:
    if name in ENSEMBLES:
        estimator = ENSEMBLES[name](cv=cv)
    elif name in BASE_MODELS:
        estimator = BASE_MODELS[name]()
    else:
        raise KeyError(f"unknown model {name!r}; available: {sorted(BASE_MODELS) + sorted(ENSEMBLES)}")

    return Pipeline([("prep", make_preprocessor(preprocessing, X)), ("model", estimator)])


def available_models() -> list[str]:
    return sorted(BASE_MODELS) + sorted(ENSEMBLES)
