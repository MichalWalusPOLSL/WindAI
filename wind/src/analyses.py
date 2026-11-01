"""Thesis analyses over the stored prediction folders.

Every analysis reads stored predictions, never retrains, and returns named
tables that main() writes to results/tables/. The folders are the ones the
configs write to, plus results/archive_wavelet_centered with the wavelet runs
on the old centred db4 bands. Model comparisons rest on the 27 location-years:
sign test, Wilcoxon and the year bootstrap. McNemar on hours rides along only
to show how far it overstates the evidence.
"""

from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import metrics
import priors
import stats
from dataset import load_location

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RUN = ["location", "model", "preprocessing", "features"]
BLOCK = ["location", "fold"]
MODELS = ["xgb", "gbt", "rf", "logreg", "knn"]
TREES = ["xgb", "gbt", "rf"]
CALIBRATED = ["xgb", "gbt"]
IS_DAY = "is_day ()"


def _load(folder: str, models: list[str], features: str) -> pd.DataFrame:
    """Prediction files of a results folder for the given models and feature set."""
    frames = [metrics.load_predictions(RESULTS / folder, f"*__{m}__*__{features}__h*") for m in models]
    return pd.concat(frames, ignore_index=True)


def _scores(frame: pd.DataFrame) -> pd.Series:
    """Accuracy, kappa, macro F1 and macro recall from the confusion counts.

    The values are sklearn's; sklearn spends most of its time validating the
    string labels, which makes thousands of folds slow.
    """
    labels = sorted(set(frame["y_true"].unique()) | set(frame["y_pred"].unique()))
    k = len(labels)
    true = pd.Categorical(frame["y_true"], categories=labels).codes.astype(np.intp)
    predicted = pd.Categorical(frame["y_pred"], categories=labels).codes.astype(np.intp)
    matrix = np.bincount(true * k + predicted, minlength=k * k).reshape(k, k)
    n, hits = matrix.sum(), np.diag(matrix)
    support, forecasts = matrix.sum(axis=1), matrix.sum(axis=0)
    accuracy = hits.sum() / n
    chance = support @ forecasts / n**2
    recall = np.divide(hits, support, out=np.zeros(k), where=support > 0)
    f1 = np.divide(2 * hits, support + forecasts, out=np.zeros(k), where=support + forecasts > 0)
    return pd.Series(
        {
            "n": n,
            "accuracy": accuracy,
            "kappa": (accuracy - chance) / (1 - chance),
            "macro_f1": f1.mean(),
            "macro_recall": recall.mean(),
        }
    )


def _scores_by(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.groupby(keys, observed=True).apply(_scores, include_groups=False).reset_index()


def _pair(scores: pd.DataFrame, a: str, b: str) -> dict:
    """Sign test, Wilcoxon and year bootstrap of column a against column b."""
    sign = stats.sign_test(scores, a, b)
    bootstrap = stats.bootstrap_years(scores, a, b)
    return {
        "a": a,
        "b": b,
        "blocks": len(scores[[a, b]].dropna()),
        "mean_difference": bootstrap["mean_difference"],
        "ci_low": bootstrap["ci_low"],
        "ci_high": bootstrap["ci_high"],
        "p_greater": bootstrap["p_greater"],
        "wins_a": sign["wins_a"],
        "wins_b": sign["wins_b"],
        "sign_p": sign["p_value"],
        "wilcoxon_p": stats.wilcoxon(scores, a, b)["p_value"],
    }


def _with_is_day(frame: pd.DataFrame) -> pd.DataFrame:
    """Join the is_day flag of each predicted hour from the location's data."""
    flags = []
    for location in frame["location"].unique():
        data = load_location(ROOT / "data" / f"{location}.csv", location=location)
        flags.append(
            pd.DataFrame(
                {
                    "location": location,
                    "timestamp": data.timestamps.astype(frame["timestamp"].dtype),
                    "is_day": data.X[IS_DAY].astype(str).map({"0": "night", "1": "day"}),
                }
            )
        )
    return frame.merge(pd.concat(flags), on=["location", "timestamp"], how="left", validate="many_to_one")


def wavelet_lookahead() -> dict[str, pd.DataFrame]:
    """What the wavelet bands were worth with and without looking ahead.

    One row per location, model and preprocessing with the scores on the base
    features, on the old centred db4 bands and on the causal Haar bands:
    centred minus causal is what looking up to 60 h ahead was worth, causal
    minus base what the bands genuinely add. The tests compare the variants
    per model and preprocessing on the 27 location-years.
    """
    predictions = pd.concat(
        [
            metrics.load_predictions(RESULTS / "predictions", "*__base__h0__loyo"),
            metrics.load_predictions(RESULTS / "predictions", "*__wavelet__h0__loyo").assign(features="causal"),
            metrics.load_predictions(RESULTS / "archive_wavelet_centered").assign(features="centred"),
        ],
        ignore_index=True,
    )
    runs = _scores_by(predictions, RUN).pivot_table(
        index=["location", "model", "preprocessing"], columns="features", values=["kappa", "macro_f1"]
    )
    runs.columns = [f"{metric}_{features}" for metric, features in runs.columns]
    for metric in ("kappa", "macro_f1"):
        runs[f"{metric}_lookahead"] = runs[f"{metric}_centred"] - runs[f"{metric}_causal"]
        runs[f"{metric}_gain"] = runs[f"{metric}_causal"] - runs[f"{metric}_base"]

    folds = _scores_by(predictions, RUN + ["fold"])
    tests = []
    for (model, preprocessing), group in folds.groupby(["model", "preprocessing"]):
        scores = group.pivot_table(index=BLOCK, columns="features", values="kappa")
        for a, b in [("causal", "base"), ("centred", "base"), ("centred", "causal")]:
            tests.append({"model": model, "preprocessing": preprocessing} | _pair(scores, a, b))
    return {"": runs.reset_index(), "_tests": pd.DataFrame(tests)}


def recall_frequency() -> dict[str, pd.DataFrame]:
    """Correlation of class recall with class share, before and after the
    training priors are taken out of the decision.

    The uniform correction gives every class the same prior, so whatever
    correlation survives it is owed to how much the classes overlap, not to how
    often they occur. Pooled rows cover the three locations (18 classes), the
    others one location (6 classes).
    """
    predictions = pd.concat([_load("predictions", CALIBRATED, f) for f in ("base", "wavelet")], ignore_index=True)
    corrected = pd.concat([priors.apply(predictions, m) for m in ("none", "uniform")], ignore_index=True)
    classes = metrics.per_class(corrected)
    rows = []
    variant = ["model", "preprocessing", "features", "prior_correction"]
    for keys, group in classes.groupby(variant):
        rows.append(dict(zip(variant, keys)) | {"scope": "pooled"} | metrics.frequency_recall_correlation(group))
        for location, single in group.groupby("location"):
            rows.append(dict(zip(variant, keys)) | {"scope": location} | metrics.frequency_recall_correlation(single))
    return {"": pd.DataFrame(rows), "_classes": classes}


def balancing() -> dict[str, pd.DataFrame]:
    """Class weighting next to the prior corrections of the unweighted runs.

    If inverse-frequency weights only shift the decision threshold, they raise
    the recall of the rare classes, lower their precision, push their frequency
    bias above 1 and leave both AUCs where they were, as the uniform correction
    of the unweighted model does. Prior corrections only for xgb and gbt, whose
    probabilities are calibrated well enough to be rescaled.
    """
    plain = _load("predictions", TREES, "base")
    plain = plain[plain["preprocessing"] == "raw"]
    calibrated = plain[plain["model"].isin(CALIBRATED)]
    frame = pd.concat(
        [
            priors.apply(plain, "none"),
            priors.apply(_load("balanced", TREES, "base"), "none"),
            priors.apply(calibrated, "uniform"),
            priors.apply(calibrated, "oracle"),
        ],
        ignore_index=True,
    )
    return {"": metrics.summarise(frame), "_classes": metrics.per_class(frame)}


def transfer() -> dict[str, pd.DataFrame]:
    """Transfer loss split into a part owed to priors and a part owed to the
    weather-to-wind relationship.

    For every source-target cell: the raw score, the scores after the oracle
    and EM prior corrections, and the target's own LOYO score from the
    diagonal. The oracle closes the part of the gap owed to the target's other
    class frequencies; what remains is owed to a different relationship.
    """
    predictions = metrics.load_predictions(RESULTS / "transfer")
    frame = pd.concat([priors.apply(predictions, m) for m in ("none", "oracle", "em")], ignore_index=True)
    runs = metrics.summarise(frame)
    classes = metrics.per_class(frame)

    measures = ["kappa", "macro_f1", "auc_ovo"]
    cells = runs.pivot_table(index=["source", "location"], columns="prior_correction", values=measures)
    cells.columns = [f"{measure}_{correction}" for measure, correction in cells.columns]
    own = runs[(runs["source"] == runs["location"]) & (runs["prior_correction"] == "none")].set_index("location")
    targets = cells.index.get_level_values("location")
    for measure in measures:
        cells[f"{measure}_diagonal"] = own[measure].reindex(targets).to_numpy()
        cells[f"{measure}_loss"] = cells[f"{measure}_diagonal"] - cells[f"{measure}_none"]
        cells[f"{measure}_loss_priors"] = cells[f"{measure}_oracle"] - cells[f"{measure}_none"]
        cells[f"{measure}_loss_relationship"] = cells[f"{measure}_diagonal"] - cells[f"{measure}_oracle"]

    bias = classes[classes["prior_correction"] == "none"].pivot_table(
        index=["source", "location"], columns="label", values="frequency_bias"
    )
    return {"": cells.reset_index(), "_bias": bias.reset_index(), "_runs": runs, "_classes": classes}


def blh_ablation() -> dict[str, pd.DataFrame]:
    """What the boundary layer height adds, by day and by night.

    ERA5 diagnoses the BLH from a bulk Richardson number with the squared model
    wind in its denominator. The stable night-time boundary layer is where that
    ties the BLH most closely to the wind, so the drop without it is expected
    to be largest at night.
    """
    full = pd.concat([_load("predictions", TREES, f) for f in ("base", "wavelet")], ignore_index=True)
    full = full[full["preprocessing"] == "raw"]
    frame = _with_is_day(pd.concat([full, _load("ablation_blh", TREES, "*noblh")], ignore_index=True))
    frame = pd.concat([frame, frame.assign(is_day="all")], ignore_index=True)

    keys = ["location", "model", "is_day"]
    runs = _scores_by(frame, RUN + ["is_day"]).pivot_table(index=keys, columns="features", values=["kappa", "macro_f1"])
    runs.columns = [f"{metric}_{features}" for metric, features in runs.columns]
    classes = metrics.per_class(frame, by=RUN + ["is_day"]).pivot_table(
        index=keys + ["label"], columns="features", values=["recall", "share"]
    )
    classes.columns = [f"{metric}_{features}" for metric, features in classes.columns]
    for table, measures in ((runs, ("kappa", "macro_f1")), (classes, ("recall",))):
        for measure in measures:
            table[f"{measure}_drop_base"] = table[f"{measure}_base"] - table[f"{measure}_noblh"]
            table[f"{measure}_drop_wavelet"] = table[f"{measure}_wavelet"] - table[f"{measure}_wavelet_noblh"]
    return {"": runs.reset_index(), "_classes": classes.reset_index()}


def horizons() -> dict[str, pd.DataFrame]:
    """Scores of xgb on the base features from t+0 to t+6 h, t+0 being the
    dashboard's LOYO run."""
    now = metrics.load_predictions(RESULTS / "predictions", "*__xgb__raw__base__h0__loyo")
    runs = pd.concat(
        [metrics.summarise(now), metrics.summarise(metrics.load_predictions(RESULTS / "horizons"))],
        ignore_index=True,
    )
    return {"": runs.sort_values(["location", "horizon"])}


def quantile() -> dict[str, pd.DataFrame]:
    """Equal-frequency classes: every prior is 1/6, so differences in recall
    between classes are owed to overlap alone."""
    predictions = metrics.load_predictions(RESULTS / "quantile")
    return {"": metrics.summarise(predictions), "_classes": metrics.per_class(predictions)}


def model_tests() -> dict[str, pd.DataFrame]:
    """Every pair of models within a preprocessing variant and feature set.

    Each pair is tested on the 27 location-years, with Holm's correction over
    the ten pairs of a variant. McNemar's p on hours, one per location, is
    there for contrast only: it treats 78 888 persistent hours as independent.
    """
    rows = []
    for features in ("base", "wavelet"):
        predictions = metrics.load_predictions(RESULTS / "predictions", f"*__{features}__h0__loyo")
        folds = _scores_by(predictions, RUN + ["fold"])
        for preprocessing, group in folds.groupby("preprocessing"):
            scores = group.pivot_table(index=BLOCK, columns="model", values="kappa")
            subset = predictions[predictions["preprocessing"] == preprocessing]
            family = []
            for a, b in combinations([m for m in MODELS if m in scores], 2):
                row = {"features": features, "preprocessing": preprocessing} | _pair(scores, a, b)
                for location in sorted(subset["location"].unique()):
                    row[f"mcnemar_p_{location}"] = stats.mcnemar(subset, a, b, location=location)["p_value"]
                family.append(row)
            family = pd.DataFrame(family)
            family["sign_p_holm"] = stats._holm(family["sign_p"].to_numpy())
            family["wilcoxon_p_holm"] = stats._holm(family["wilcoxon_p"].to_numpy())
            rows.append(family)
    return {"": pd.concat(rows, ignore_index=True)}


def device_noise(cpu: str | Path, cuda: str | Path = RESULTS / "predictions") -> pd.DataFrame:
    """CPU minus CUDA scores of the same xgb runs, the floor of numerical noise.

    xgboost's CPU and CUDA tree builders are different implementations. In
    most folds the two fits agree up to float32 rounding and no hour changes
    class; in a few, a split decided differently early on sends the ensembles
    apart and a tenth of the hours change class. All xgb runs are fitted on
    CUDA; a difference between two runs no larger than this one cannot be told
    apart from the device. One row per location and fold, one pooled row per
    location (fold "all") and the threshold row (location "all", fold
    "max_abs"): the largest absolute difference over all location-years.
    """
    rows = []
    for path in sorted(Path(cpu).glob("*__xgb__*.parquet")):
        on_cpu, on_cuda = pd.read_parquet(path), pd.read_parquet(Path(cuda) / path.name)
        if not np.array_equal(on_cpu["timestamp"].to_numpy(), on_cuda["timestamp"].to_numpy()):
            raise ValueError(f"{path.name}: the CPU and CUDA runs do not cover the same hours")
        location = on_cpu["location"].iat[0]
        folds = [(fold, on_cpu["fold"] == fold) for fold in sorted(on_cpu["fold"].unique())]
        for fold, mask in folds + [("all", pd.Series(True, index=on_cpu.index))]:
            a, b = _scores(on_cpu[mask]), _scores(on_cuda[mask])
            changed = on_cpu.loc[mask, "y_pred"].to_numpy() != on_cuda.loc[mask, "y_pred"].to_numpy()
            years = on_cpu.loc[mask, "timestamp"].dt.year.unique()
            rows.append(
                {
                    "location": location,
                    "fold": fold,
                    "year": years[0] if len(years) == 1 else None,
                    "n": int(mask.sum()),
                    "changed_hours": int(changed.sum()),
                    "changed_share": changed.mean(),
                    "kappa_cpu": a["kappa"],
                    "kappa_cuda": b["kappa"],
                    "kappa_diff": a["kappa"] - b["kappa"],
                    "macro_f1_cpu": a["macro_f1"],
                    "macro_f1_cuda": b["macro_f1"],
                    "macro_f1_diff": a["macro_f1"] - b["macro_f1"],
                }
            )
    table = pd.DataFrame(rows)
    folds = table[table["fold"] != "all"]
    threshold = {
        "location": "all",
        "fold": "max_abs",
        "changed_share": folds["changed_share"].max(),
        "kappa_diff": folds["kappa_diff"].abs().max(),
        "macro_f1_diff": folds["macro_f1_diff"].abs().max(),
    }
    out = pd.concat([table, pd.DataFrame([threshold])], ignore_index=True)
    return out.astype({"year": "Int64", "n": "Int64", "changed_hours": "Int64"})


ANALYSES = {
    "wavelet_lookahead": wavelet_lookahead,
    "recall_frequency": recall_frequency,
    "balancing": balancing,
    "transfer": transfer,
    "blh_ablation": blh_ablation,
    "horizons": horizons,
    "quantile": quantile,
    "model_tests": model_tests,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(RESULTS / "tables"))
    parser.add_argument("--only", nargs="*", choices=sorted(ANALYSES), help="run only these analyses")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, analysis in ANALYSES.items():
        if args.only and name not in args.only:
            continue
        started = time.perf_counter()
        for suffix, table in analysis().items():
            table.to_csv(out / f"{name}{suffix}.csv", index=False)
            print(f"  {name}{suffix}.csv  {len(table)} rows", flush=True)
        print(f"{name}: {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
