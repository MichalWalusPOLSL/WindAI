"""Exports the tables needed to review a finished experiment grid."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import metrics  # noqa: E402

OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

predictions = metrics.load_predictions(ROOT / "results" / "predictions")
print(f"{len(predictions):,} rows, {predictions.groupby(metrics.ID_COLUMNS).ngroups} runs")

summary = metrics.summarise(predictions)
summary["lift"] = summary["accuracy"] - summary["zero_r"]
summary.to_csv(OUT / "summary.csv", index=False)

per_class = metrics.per_class(predictions)
per_class.to_csv(OUT / "per_class.csv", index=False)

folds = (
    predictions.groupby(metrics.ID_COLUMNS + ["fold"])
    .apply(lambda g: pd.Series({"kappa": cohen_kappa_score(g.y_true, g.y_pred)}),
           include_groups=False)
    .reset_index()
)
folds.to_csv(OUT / "folds.csv", index=False)

for name in ("summary", "per_class", "folds"):
    path = OUT / f"{name}.csv"
    print(f"  {path.name}: {path.stat().st_size / 1024:.0f} kB")