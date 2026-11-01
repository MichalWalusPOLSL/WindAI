"""Dataset loaders for two sources of the same experiment.

load_prepared reads a table already preprocessed in Altair AI Studio (columns
dropped, weather_code/is_day nominal, beaufort label present) and only derives
the lag features AI Studio cannot express. load_raw does the whole preparation
in Python from an untouched Open-Meteo export, and is kept so the two paths can
be cross-checked against each other. Only load_raw still sees the wind speed,
so only it can label the target with a class scheme other than Beaufort.
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
QUANTILE_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
CLASS_SCHEMES = ("beaufort", "quantile")

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
WAVELET_LEVEL = 4
WAVELET_SOURCES = (PRESSURE, PBL)
FEATURE_SETS = ("base", "wavelet", "noblh", "wavelet_noblh")


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


def _haar_a_trous(values: np.ndarray) -> np.ndarray:
    """Causal redundant Haar transform (Renaud, Starck and Murtagh 2005).

    With c_0 the series itself, c_{j+1}(t) = (c_j(t - 2^j) + c_j(t)) / 2 and
    w_{j+1} = c_j - c_{j+1}, where t - 2^j before the first hour falls back to
    the first value. Each coefficient at hour t is built from hours up to t
    only, at most 2^J - 1 = 15 h back. The db4 SWT used before was centred: an
    impulse showed its bands at hour t reaching up to 60 h ahead. Returns the
    columns w_1..w_J, c_J, which sum back to the input.
    """
    smooth = np.asarray(values, dtype=float)
    hours = np.arange(len(smooth))
    bands = []
    for level in range(WAVELET_LEVEL):
        coarser = 0.5 * (smooth[np.maximum(hours - 2**level, 0)] + smooth)
        bands.append(smooth - coarser)
        smooth = coarser
    return np.column_stack(bands + [smooth])


def _band_names(column: str) -> list[str]:
    short = column.split(" ")[0]
    details = [f"{short}_w{j}" for j in range(1, WAVELET_LEVEL + 1)]
    return details + [f"{short}_c{WAVELET_LEVEL}"]


def _wavelet_bands(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal wavelet bands of the predictor series over the whole record.

    Decomposing the target would leak it back in shredded form, so only
    predictors are transformed. The transform never looks ahead, so a single
    pass over the record is safe: like the tendency lags, the first hours of a
    year read the last hours of the year before, and no hour sees anything
    later than itself.
    """
    missing = [c for c in WAVELET_SOURCES if c not in frame]
    if missing:
        raise KeyError(f"cannot build wavelet bands, missing: {missing}")

    out = {}
    for column in WAVELET_SOURCES:
        bands = _haar_a_trous(frame[column].to_numpy(dtype=float))
        out |= dict(zip(_band_names(column), bands.T))
    return pd.DataFrame(out, index=frame.index)


def _pbl_columns() -> list[str]:
    return [PBL, *(f"dblh_{lag}h (m)" for lag in PBL_LAGS), *_band_names(PBL)]


def _feature_set(X: pd.DataFrame, frame: pd.DataFrame, features: str) -> pd.DataFrame:
    """Add the wavelet bands and drop the BLH predictors the feature set names.

    Sets starting with "wavelet" add the bands. Sets ending with "noblh" drop
    the boundary layer height together with its tendency and its bands, and
    keep the pressure bands: ERA5 diagnoses the BLH from a bulk Richardson
    number with the squared model wind in its denominator, so the strongest
    predictor may carry the target back in.
    """
    if features.startswith("wavelet"):
        X = pd.concat([X, _wavelet_bands(frame)], axis=1)
    if features.endswith("noblh"):
        X = X.drop(columns=[c for c in _pbl_columns() if c in X])
    return X


def _target(wind: pd.Series, classes: str, name: str) -> pd.Series:
    """Label the wind speed with a class scheme.

    beaufort cuts at the fixed scale thresholds. quantile cuts at the sextiles
    of the location's whole record, so every class holds a sixth of the hours.
    The cut points belong to the definition of the label, like the Beaufort
    thresholds, and are not refitted inside a fold.
    """
    if classes == "beaufort":
        target = pd.cut(wind, bins=BEAUFORT_BINS, labels=BEAUFORT_LABELS, right=False)
    else:
        target = pd.qcut(wind, len(QUANTILE_LABELS), labels=QUANTILE_LABELS)
    if target.isna().any():
        raise ValueError(f"{name}: {int(target.isna().sum())} rows outside the {classes} classes")
    return target


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
    features: str = "base",
    horizon: int = 0,
    classes: str = "beaufort",
) -> Dataset:
    path = Path(path)
    if classes != "beaufort":
        raise ValueError(
            f"{path.name}: classes={classes!r} needs the wind speed, which the AI Studio "
            "export no longer has; use source: raw"
        )
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
    X = _feature_set(X, frame, features)

    for column in X.columns:
        if X[column].dtype == object or column in CATEGORICAL:
            X[column] = X[column].astype(str).str.strip().astype("category")

    return _finalise(X, y, timestamps, location or path.stem, horizon)


def load_raw(
    path: str | Path,
    location: str | None = None,
    add_tendency: bool = True,
    features: str = "base",
    horizon: int = 0,
    classes: str = "beaufort",
) -> Dataset:
    path = Path(path)
    frame = _read(path)
    timestamps = pd.to_datetime(frame[TIME])
    target = _target(frame[TARGET], classes, path.name)

    X = frame.drop(columns=[TIME, TARGET])
    if add_tendency:
        X = pd.concat([X, _tendency(frame)], axis=1)
    X = _feature_set(X, frame, features)

    X = X.drop(columns=[c for c in DROP_ALWAYS if c in X], errors="ignore")
    for column in CATEGORICAL:
        if column in X:
            X[column] = X[column].astype("category")

    return _finalise(X, target.astype(str).to_numpy(), timestamps, location or path.stem, horizon)


LOADERS = {"prepared": load_prepared, "raw": load_raw}


def load_location(path, source: str = "prepared", **kwargs) -> Dataset:
    if source not in LOADERS:
        raise KeyError(f"unknown source {source!r}; available: {sorted(LOADERS)}")

    features = kwargs.get("features", "base")
    if features not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {features!r}; available: {sorted(FEATURE_SETS)}")

    classes = kwargs.get("classes", "beaufort")
    if classes not in CLASS_SCHEMES:
        raise KeyError(f"unknown class scheme {classes!r}; available: {sorted(CLASS_SCHEMES)}")

    return LOADERS[source](path, **kwargs)


def class_labels(labels) -> list[str]:
    """The ordered label list of the scheme the given labels belong to.

    Beaufort and quantile labels come back as the whole scale in its order,
    absent classes included; anything else is sorted. This is also the order
    the ordinal metrics count class distances in.
    """
    present = {str(label) for label in labels}
    for scheme in (BEAUFORT_LABELS, QUANTILE_LABELS):
        if present <= set(scheme):
            return list(scheme)
    return sorted(present)


def class_counts(dataset: Dataset) -> pd.Series:
    labels = class_labels(dataset.y)
    return pd.Series(dataset.y).value_counts().reindex(labels).fillna(0).astype(int)
