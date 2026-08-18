"""Dataset loaders for two sources of the same experiment.

load_prepared reads a table already preprocessed in Altair AI Studio (columns
dropped, weather_code/is_day nominal, beaufort label present) and only derives
the lag features AI Studio cannot express. load_raw does the whole preparation
in Python from an untouched Open-Meteo export, and is kept so the two paths can
be cross-checked against each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIME = "time"
LABEL = "beaufort"
TARGET = "wind_speed_100m (m/s)"

BEAUFORT_BINS = [-np.inf, 1.6, 3.4, 5.5, 8.0, 10.8, np.inf]
BEAUFORT_LABELS = ["Bf0-1", "Bf2", "Bf3", "Bf4", "Bf5", "Bf6+"]

DROP_ALWAYS = [
    "snow_depth (m)",
    "surface_pressure (hPa)",
    "precipitation (mm)",
    "snowfall (cm)",
]
CATEGORICAL = ["weather_code (wmo code)", "is_day ()"]

PRESSURE = "pressure_msl (hPa)"
PBL = "boundary_layer_height (m)"
PRESSURE_LAGS = (3, 6, 24)
PBL_LAGS = (3,)
ENCODINGS = ("utf-8", "cp1252")


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    years: np.ndarray
    timestamps: pd.Series
    location: str

    def __len__(self) -> int:
        return len(self.y)


def _read(path: Path) -> pd.DataFrame:
    """Sniff separator and encoding.

    The AI Studio exports are semicolon separated and cp1252, not utf-8: the
    degree sign in the temperature column names is a single 0xb0 byte.
    """
    for encoding in ENCODINGS:
        for separator in (",", ";"):
            try:
                frame = pd.read_csv(
                    path,
                    sep=separator,
                    comment="#",
                    skipinitialspace=True,
                    encoding=encoding,
                )
            except UnicodeDecodeError:
                break
            if frame.shape[1] > 1:
                frame.columns = [c.strip().strip('"') for c in frame.columns]
                return frame
    raise ValueError(f"{path.name}: could not determine column separator or encoding")


def _tendency(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in (PRESSURE, PBL) if c not in frame]
    if missing:
        raise KeyError(f"cannot build lag features, missing: {missing}")
    out = {f"dp_{lag}h (hPa)": frame[PRESSURE].diff(lag) for lag in PRESSURE_LAGS}
    out |= {f"dblh_{lag}h (m)": frame[PBL].diff(lag) for lag in PBL_LAGS}
    return pd.DataFrame(out, index=frame.index).bfill()


def _regular_hours(timestamps: pd.Series, location: str) -> pd.Series:
    """Repair the DST artefacts of the export, reject anything worse.

    The export carries 24 rows for every day, the two switch days included, so
    the row order is a contiguous hourly series while the labels skip 02:00 in
    March and repeat the next hour. Rebuilding the index from the first label
    puts the series back on one fixed offset. Nothing downstream reads a wall
    clock: timestamps only key the folds and join the prediction files, and the
    lag features in _tendency count rows, not labels.
    """
    irregular = int((timestamps.diff().dropna() != pd.Timedelta("1h")).sum())
    if not irregular:
        return timestamps
    if irregular > 2 * timestamps.dt.year.nunique():
        raise ValueError(f"{location}: {irregular} irregular time steps, more than DST explains")
    print(f"  note: {location}: rebuilt {irregular} DST-mangled timestamps", flush=True)
    return pd.Series(pd.date_range(timestamps.iloc[0], periods=len(timestamps), freq="h"))


def _finalise(X, y, timestamps, location, horizon) -> Dataset:
    timestamps = _regular_hours(timestamps, location)
    if horizon:
        X = X.iloc[:-horizon].reset_index(drop=True)
        y = y[horizon:]
        timestamps = timestamps.iloc[horizon:]
    return Dataset(
        X=X.reset_index(drop=True),
        y=np.asarray(y),
        years=timestamps.dt.year.to_numpy(),
        timestamps=timestamps.reset_index(drop=True),
        location=location,
    )


def load_prepared(
    path: str | Path,
    location: str | None = None,
    add_tendency: bool = True,
    horizon: int = 0,
) -> Dataset:
    path = Path(path)
    frame = _read(path)

    if LABEL not in frame:
        raise KeyError(f"{path.name}: no {LABEL!r} column; export the table after Set Role")
    if TARGET in frame:
        raise ValueError(f"{path.name}: {TARGET!r} is still present, this leaks the label")

    timestamps = pd.to_datetime(frame[TIME])
    y = frame[LABEL].astype(str).str.strip().to_numpy()
    X = frame.drop(columns=[TIME, LABEL])

    if add_tendency:
        X = pd.concat([X, _tendency(frame)], axis=1)

    for column in X.columns:
        if X[column].dtype == object or column in CATEGORICAL:
            X[column] = X[column].astype(str).str.strip().astype("category")

    return _finalise(X, y, timestamps, location or path.stem, horizon)


def load_raw(
    path: str | Path,
    location: str | None = None,
    add_tendency: bool = True,
    horizon: int = 0,
) -> Dataset:
    path = Path(path)
    frame = _read(path)
    timestamps = pd.to_datetime(frame[TIME])

    target = pd.cut(frame[TARGET], bins=BEAUFORT_BINS, labels=BEAUFORT_LABELS, right=False)
    if target.isna().any():
        raise ValueError(f"{path.name}: {int(target.isna().sum())} rows outside Beaufort bins")

    X = frame.drop(columns=[TIME, TARGET])
    if add_tendency:
        X = pd.concat([X, _tendency(frame)], axis=1)
    X = X.drop(columns=[c for c in DROP_ALWAYS if c in X], errors="ignore")
    for column in CATEGORICAL:
        if column in X:
            X[column] = X[column].astype("category")

    return _finalise(X, target.astype(str).to_numpy(), timestamps, location or path.stem, horizon)


LOADERS = {"prepared": load_prepared, "raw": load_raw}


def load_location(path, source: str = "prepared", **kwargs) -> Dataset:
    if source not in LOADERS:
        raise KeyError(f"unknown source {source!r}; available: {sorted(LOADERS)}")
    return LOADERS[source](path, **kwargs)


def class_counts(dataset: Dataset) -> pd.Series:
    return pd.Series(dataset.y).value_counts().reindex(BEAUFORT_LABELS).fillna(0).astype(int)
