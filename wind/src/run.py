"""Experiment runner: iterates over the configuration grid and stores predictions.

Each combination of location, model, preprocessing and horizon produces one
prediction file containing per-fold out-of-fold predictions and class
probabilities. Metrics are never computed here.
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
from sklearn.base import clone

from dataset import load_location
from registry import SCALE_SENSITIVE, build_model
from splits import build_splitter

ROOT = Path(__file__).resolve().parent.parent


def _write(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except (ImportError, ValueError):
        target = path.with_suffix(".csv.gz")
        frame.to_csv(target, index=False, compression="gzip")
        return target


def run_one(dataset, model_name, preprocessing, splitter_name, horizon, outdir) -> Path:
    splitter = build_splitter(splitter_name, dataset.years)
    inner = build_splitter(splitter_name, dataset.years)
    template = build_model(model_name, dataset.X, preprocessing, cv=inner)

    if model_name in SCALE_SENSITIVE and preprocessing == "raw":
        print(f"  note: {model_name} is scale sensitive, consider preprocessing='scaled'", flush=True)

    rows = []
    started = time.perf_counter()
    for fold, (train, test) in enumerate(splitter.split(dataset.X, dataset.y), start=1):
        estimator = clone(template)
        estimator.fit(dataset.X.iloc[train], dataset.y[train])
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
        rows.append(block)

    frame = pd.concat(rows, ignore_index=True)
    frame.insert(0, "location", dataset.location)
    frame.insert(1, "model", model_name)
    frame.insert(2, "preprocessing", preprocessing)
    frame.insert(3, "horizon", horizon)
    frame.insert(4, "splitter", splitter_name)

    stem = f"{dataset.location}__{model_name}__{preprocessing}__h{horizon}__{splitter_name}"
    path = _write(frame, outdir / stem)
    elapsed = time.perf_counter() - started
    accuracy = float((frame["y_pred"] == frame["y_true"]).mean())
    print(f"  {stem}  acc={accuracy:.4f}  {elapsed:.1f}s -> {path.name}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_dir = (ROOT / config["data_dir"]).resolve()
    outdir = (ROOT / config["predictions_dir"]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    grid = list(
        product(
            config["locations"],
            config["models"],
            config["preprocessing"],
            config["horizons"],
            config["splitters"],
        )
    )
    print(f"{len(grid)} runs -> {outdir}")

    cache: dict[tuple[str, int], object] = {}
    manifest = []
    for location, model_name, preprocessing, horizon, splitter_name in grid:
        stem = f"{location}__{model_name}__{preprocessing}__h{horizon}__{splitter_name}"
        existing = list(outdir.glob(stem + ".*"))
        if existing and not args.overwrite:
            print(f"  {stem}  skipped")
            continue

        key = (location, horizon)
        if key not in cache:
            cache[key] = load_location(
                data_dir / f"{location}.csv",
                source=config.get("source", "prepared"),
                location=location,
                add_tendency=config.get("add_tendency", True),
                horizon=horizon,
            )
        path = run_one(cache[key], model_name, preprocessing, splitter_name, horizon, outdir)
        manifest.append({"stem": stem, "file": path.name})

    if manifest:
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
