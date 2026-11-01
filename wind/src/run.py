"""Experiment runner: iterates over the configuration grid and stores predictions.

Each combination of location, model, preprocessing, feature set, horizon,
splitter, class weighting and class scheme produces one prediction file
containing per-fold out-of-fold predictions, class probabilities and the class
priors of the fold's training set. Metrics are never computed here.

With mode: transfer the splitters give way to ordered location pairs: for every
year both locations cover, the model trains on the source without that year
and predicts the target in it. A location paired with itself is LOYO computed
through the same code path.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import has_fit_parameter

from dataset import load_location
from registry import SCALE_SENSITIVE, build_model
from splits import build_splitter

ROOT = Path(__file__).resolve().parent.parent
MODES = ("grid", "transfer")
TRANSFER = "transfer"
BALANCING = ("none", "balanced")
DEFAULTS = {"balancing": "none", "classes": "beaufort"}


def _write(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except (ImportError, ValueError):
        target = path.with_suffix(".csv.gz")
        frame.to_csv(target, index=False, compression="gzip")
        return target


def _stem(location, model_name, preprocessing, features, horizon, splitter_name, balancing, classes) -> str:
    """File name of a run without the extension.

    Default balancing and classes add no suffix, so every file written before
    the two options existed keeps its name and is still skipped.
    """
    stem = f"{location}__{model_name}__{preprocessing}__{features}__h{horizon}__{splitter_name}"
    if balancing != DEFAULTS["balancing"]:
        stem += f"__bal-{balancing}"
    if classes != DEFAULTS["classes"]:
        stem += f"__cls-{classes}"
    return stem


def _folds(splitter_name, source, target):
    """Yield (fold, training rows of source, test rows of target).

    A transfer fold is one year both locations cover. The year is held out of
    the source as well: the locations sit under the same synoptic systems, so
    training on the source in the test year would hand the model its weather.
    """
    if splitter_name == TRANSFER:
        for fold, year in enumerate(np.intersect1d(source.years, target.years), start=1):
            yield fold, np.flatnonzero(source.years != year), np.flatnonzero(target.years == year)
        return
    splitter = build_splitter(splitter_name, target.years)
    for fold, (train, test) in enumerate(splitter.split(target.X, target.y), start=1):
        yield fold, train, test


def _fit(estimator, X, y, model_name, balancing):
    """Fit the pipeline, with sample weights that balance the classes if asked."""
    if balancing == "none":
        return estimator.fit(X, y)
    if not has_fit_parameter(estimator[-1], "sample_weight"):
        raise ValueError(
            f"{model_name} does not accept sample_weight, so balancing={balancing!r} "
            "cannot reach it; remove the model from this grid"
        )
    return estimator.fit(X, y, model__sample_weight=compute_sample_weight(balancing, y))


def run_one(
    dataset,
    model_name,
    preprocessing,
    features,
    splitter_name,
    horizon,
    outdir,
    balancing="none",
    classes="beaufort",
    source=None,
) -> Path:
    """Train and predict every fold of one run and write its prediction file.

    dataset is the location predicted. source is the location trained on; it
    defaults to dataset, and any other location needs the transfer splitter.
    The model is rebuilt in every fold, with the inner splitter of an ensemble
    built on that fold's training years, so its indices address the rows the
    ensemble is actually fitted on.
    """
    source = dataset if source is None else source
    if splitter_name != TRANSFER and source is not dataset:
        raise ValueError(f"training on {source.location} to predict {dataset.location} needs the transfer splitter")
    if list(source.X.columns) != list(dataset.X.columns):
        raise ValueError(f"{source.location} and {dataset.location} do not have the same feature columns")
    inner_name = "loyo" if splitter_name == TRANSFER else splitter_name

    if model_name in SCALE_SENSITIVE and preprocessing == "raw":
        print(f"  note: {model_name} is scale sensitive, consider preprocessing='scaled'", flush=True)

    rows = []
    started = time.perf_counter()
    for fold, train, test in _folds(splitter_name, source, dataset):
        inner = build_splitter(inner_name, source.years[train])
        estimator = build_model(model_name, source.X, preprocessing, cv=inner)
        _fit(estimator, source.X.iloc[train], source.y[train], model_name, balancing)
        predicted = estimator.predict(dataset.X.iloc[test])

        block = pd.DataFrame(
            {
                "timestamp": dataset.timestamps.iloc[test].to_numpy(),
                "fold": fold,
                "y_true": dataset.y[test],
                "y_pred": predicted,
            }
        )
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(dataset.X.iloc[test])
            for i, label in enumerate(estimator.classes_):
                block[f"proba_{label}"] = proba[:, i]
        labels, counts = np.unique(source.y[train], return_counts=True)
        for label, count in zip(labels, counts):
            block[f"prior_{label}"] = count / len(train)
        rows.append(block)

    frame = pd.concat(rows, ignore_index=True)
    distributions = [c for c in frame.columns if c.startswith(("proba_", "prior_"))]
    frame[distributions] = frame[distributions].fillna(0.0)

    ids = {"location": dataset.location}
    if splitter_name == TRANSFER:
        ids["source"] = source.location
    ids |= {
        "model": model_name,
        "preprocessing": preprocessing,
        "features": features,
        "horizon": horizon,
        "splitter": splitter_name,
    }
    ids |= {k: v for k, v in (("balancing", balancing), ("classes", classes)) if v != DEFAULTS[k]}
    for position, (column, value) in enumerate(ids.items()):
        frame.insert(position, column, value)

    name = f"{source.location}-to-{dataset.location}" if splitter_name == TRANSFER else dataset.location
    stem = _stem(name, model_name, preprocessing, features, horizon, splitter_name, balancing, classes)

    path = _write(frame, outdir / stem)
    elapsed = time.perf_counter() - started
    accuracy = float((frame["y_pred"] == frame["y_true"]).mean())
    print(f"  {stem}  acc={accuracy:.4f}  {elapsed:.1f}s -> {path.name}", flush=True)
    return path


def _pairs(config) -> list[tuple[str, str]]:
    """Ordered (source, target) pairs of a transfer config, the diagonal included."""
    pairs = config.get("pairs", "all")
    if pairs == "all":
        return list(product(config["locations"], repeat=2))
    if isinstance(pairs, str) or any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"pairs must be 'all' or a list of [source, target], got {pairs!r}")
    return [tuple(pair) for pair in pairs]


def _grid(config) -> list[tuple]:
    """The runs of a config as (source, target, model, preprocessing, features,
    horizon, splitter, balancing, classes); source and target coincide outside
    transfer mode."""
    mode = config.get("mode", "grid")
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}; available: {sorted(MODES)}")
    if mode == TRANSFER:
        pairs, splitters = _pairs(config), [TRANSFER]
    else:
        pairs, splitters = [(location, location) for location in config["locations"]], config["splitters"]

    balancing = config.get("balancing", [DEFAULTS["balancing"]])
    unknown = sorted(set(balancing) - set(BALANCING))
    if unknown:
        raise KeyError(f"unknown balancing {unknown}; available: {sorted(BALANCING)}")

    return [
        (source, target, *rest)
        for (source, target), *rest in product(
            pairs,
            config["models"],
            config["preprocessing"],
            config.get("features", ["base"]),
            config["horizons"],
            splitters,
            balancing,
            config.get("classes", [DEFAULTS["classes"]]),
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_dir = (ROOT / config["data_dir"]).resolve()
    outdir = (ROOT / config["predictions_dir"]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    grid = _grid(config)
    print(f"{len(grid)} runs -> {outdir}")

    cache: dict[tuple[str, str, int, str], object] = {}
    manifest = []
    for source_name, location, model_name, preprocessing, features, horizon, splitter_name, balancing, classes in grid:
        name = f"{source_name}-to-{location}" if splitter_name == TRANSFER else location
        stem = _stem(name, model_name, preprocessing, features, horizon, splitter_name, balancing, classes)
        existing = list(outdir.glob(stem + ".*"))
        if existing and not args.overwrite:
            print(f"  {stem}  skipped")
            continue

        for key in {(n, features, horizon, classes) for n in (source_name, location)}:
            if key not in cache:
                cache[key] = load_location(
                    data_dir / f"{key[0]}.csv",
                    source=config.get("source", "prepared"),
                    location=key[0],
                    add_tendency=config.get("add_tendency", True),
                    features=features,
                    horizon=horizon,
                    classes=classes,
                )
        path = run_one(
            cache[(location, features, horizon, classes)],
            model_name,
            preprocessing,
            features,
            splitter_name,
            horizon,
            outdir,
            balancing=balancing,
            classes=classes,
            source=cache[(source_name, features, horizon, classes)],
        )
        manifest.append({"stem": stem, "file": path.name})

    if manifest:
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
