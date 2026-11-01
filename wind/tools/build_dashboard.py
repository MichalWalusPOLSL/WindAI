"""Builds a standalone HTML dashboard of a finished experiment grid.

Reads every prediction file, computes the run summary, per-class scores,
per-year kappa and confusion counts, and writes them to results/dashboard/ as
CSV next to index.html: a single page with the data embedded that opens in any
browser (d3 loads from cdnjs, so the first open needs a connection). The page
itself is tools/dashboard/page.html, the same file the hosted dashboard runs;
tools/dashboard/shell.html stands in for the hosted dashboard around it.

The page compares runs of one horizon and one splitter, picked with --horizon
and --splitter. Prediction files written before the feature-set column existed
are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import metrics  # noqa: E402

TEMPLATES = Path(__file__).resolve().parent / "dashboard"
KEY = ["location", "model", "preprocessing", "features"]
PARAMS = {"location": "suwalki", "features": "wavelet", "metric": "kappa", "run": None}


def _run_id(frame: pd.DataFrame) -> pd.Series:
    return frame[KEY].astype(str).agg("|".join, axis=1)


def _tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """The four dashboard tables for one prediction file, i.e. one run."""
    summary = metrics.summarise(frame)
    per_class = metrics.per_class(frame)

    years = pd.to_datetime(frame["timestamp"]).dt.year
    folds = (
        frame.assign(year=years)
        .groupby(KEY + ["fold"], observed=True)
        .apply(
            lambda g: pd.Series(
                {"kappa": cohen_kappa_score(g.y_true, g.y_pred), "year": g.year.mode().iat[0]}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    folds["year"] = folds["year"].astype(int)

    confusion = frame.groupby(KEY + ["y_true", "y_pred"], observed=True).size().rename("n").reset_index()
    confusion["share_of_true"] = confusion["n"] / confusion.groupby("y_true")["n"].transform("sum")
    return summary, per_class, folds, confusion


def _finish(table: pd.DataFrame) -> pd.DataFrame:
    table = table.drop(columns=["horizon", "splitter"], errors="ignore")
    table.insert(0, "run", _run_id(table))
    if "n" in table:
        table["n"] = table["n"].astype(int)
    floats = table.select_dtypes("float").columns
    table[floats] = table[floats].round(4)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", default=str(ROOT / "results" / "predictions"))
    parser.add_argument("--out", default=str(ROOT / "results" / "dashboard"))
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--splitter", default="loyo")
    args = parser.parse_args()

    source = Path(args.predictions)
    files = sorted(source.glob("*.parquet")) or sorted(source.glob("*.csv.gz"))
    if not files:
        raise SystemExit(f"no prediction files in {source}")

    parts: list[list[pd.DataFrame]] = [[], [], [], []]
    legacy, other = 0, 0
    for path in files:
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if "features" not in frame or frame["features"].isna().all():
            legacy += 1
            continue
        if not ((frame["horizon"] == args.horizon) & (frame["splitter"] == args.splitter)).all():
            other += 1
            continue
        for part, table in zip(parts, _tables(frame)):
            part.append(table)
        print(f"  {path.name}", flush=True)

    if not parts[0]:
        raise SystemExit(f"no runs with horizon {args.horizon} and splitter {args.splitter!r} in {source}")

    summary, per_class, folds, confusion = (_finish(pd.concat(p, ignore_index=True)) for p in parts)
    summary = summary.sort_values("kappa", ascending=False)
    if summary["run"].duplicated().any():
        raise SystemExit("two prediction files describe the same run; remove the stale one")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"summary": summary, "per_class": per_class, "folds": folds, "confusion": confusion}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)

    data = "{" + ",".join(
        f'"{name}":{table.to_json(orient="records", force_ascii=False)}' for name, table in tables.items()
    ) + "}"
    footer = (
        f"Wygenerowano {datetime.now():%Y-%m-%d %H:%M} przez tools/build_dashboard.py z {source.name}/: "
        f"{len(summary)} przebiegów, horyzont {args.horizon} h, walidacja {args.splitter}."
    )
    page = (
        (TEMPLATES / "shell.html").read_text(encoding="utf-8")
        .replace("__PARAMS__", json.dumps(PARAMS))
        .replace("__DATA__", data.replace("</", "<\\/"))
        .replace("__FOOTER__", footer)
        .replace("__PAGE__", (TEMPLATES / "page.html").read_text(encoding="utf-8"))
    )
    (out / "index.html").write_text(page, encoding="utf-8")

    print(f"{len(summary)} runs -> {out / 'index.html'}")
    if legacy:
        print(f"  skipped {legacy} files without a feature set (written before it existed)")
    if other:
        print(f"  skipped {other} files with another horizon or splitter")


if __name__ == "__main__":
    main()
