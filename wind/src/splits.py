"""Time-aware cross-validation splitters for autocorrelated hourly series.

LeaveOneYearOut holds out one calendar year per fold; WalkForward uses an
expanding training window. Both follow the scikit-learn splitter protocol and
can be passed to any estimator, including the inner cv of StackingClassifier.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class LeaveOneYearOut:
    """One fold per calendar year; each test fold spans a full seasonal cycle."""

    def __init__(self, years: np.ndarray):
        self.years = np.asarray(years)
        self._unique = np.unique(self.years)

    def split(self, X=None, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for year in self._unique:
            test = np.flatnonzero(self.years == year)
            train = np.flatnonzero(self.years != year)
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self._unique)

    def __repr__(self) -> str:
        return f"LeaveOneYearOut(n_splits={self.get_n_splits()})"


class WalkForward:
    """Expanding window: train on all years up to k, test on year k+1."""

    def __init__(self, years: np.ndarray, min_train_years: int = 2):
        self.years = np.asarray(years)
        self.min_train_years = min_train_years
        self._unique = np.unique(self.years)

    def split(self, X=None, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for i in range(self.min_train_years, len(self._unique)):
            train = np.flatnonzero(np.isin(self.years, self._unique[:i]))
            test = np.flatnonzero(self.years == self._unique[i])
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return max(0, len(self._unique) - self.min_train_years)

    def __repr__(self) -> str:
        return f"WalkForward(n_splits={self.get_n_splits()})"


SPLITTERS = {"loyo": LeaveOneYearOut, "walk_forward": WalkForward}


def build_splitter(name: str, years: np.ndarray, **kwargs):
    if name not in SPLITTERS:
        raise KeyError(f"unknown splitter {name!r}; available: {sorted(SPLITTERS)}")
    return SPLITTERS[name](years, **kwargs)
