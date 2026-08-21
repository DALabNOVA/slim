# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Experimental campaign for the MS-SLIM manuscript.

Implements the protocol of ``MS_SLIM_research_plan.md``: identical stratified
splits across methods, 20 runs per configuration, paired statistics
with multiple-comparison correction and effect sizes, and one result row per
(question, dataset, method, seed).

Run from the command line::

    python -m slim_gsgp.classification.campaign --question q1 --datasets spambase
    python -m slim_gsgp.classification.campaign --question all --out results/

Each question writes ``<out>/<question>.csv``; ``analyse`` turns those into the
paired comparison tables the manuscript reports.
"""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)

from slim_gsgp.classification.benchmarks import (
    BINARY_DATASETS, MULTICLASS_DATASETS, load_dataset_split,
)
from slim_gsgp.classification.codes import simplex_codes
from slim_gsgp.classification.strategies import get_strategy
from slim_gsgp.classification.calibration import (
    binary_probabilities,
    multiclass_temperature_probabilities,
)

__all__ = [
    "evaluate_predictions", "run_binary_config",
    "run_multiclass_config", "run_question", "paired_comparison", "analyse",
]

# Budgets. The plan requires budgets matched across methods, not just population
# sizes, so every question below holds pop_size x n_iter fixed and varies only
# the factor under study.
DEFAULT_POP_SIZE = 100
DEFAULT_N_ITER = 100
DEFAULT_SEEDS = 20
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
MUTATION_STEPS = (0.1, 0.3, 0.5, 1.0, 2.0)
def evaluate_predictions(y_true, y_pred, scores=None, n_classes=2) -> dict:
    """
    Metrics required by the research plan's evaluation section.

    Binary problems additionally report AUROC and AUPRC from raw scores;
    multiclass problems report macro-F1 in place of binary F1.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted class codes.
    scores : array-like, optional
        Raw decision scores for the positive class (binary only).
    n_classes : int, optional
        Number of classes.

    Returns
    -------
    dict
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if n_classes == 2:
        metrics["f1"] = f1_score(y_true, y_pred, pos_label=y_true.max(), zero_division=0)
        if scores is not None:
            positive = (y_true == y_true.max())
            # A degenerate split with one class present makes both undefined.
            if positive.any() and not positive.all():
                metrics["auroc"] = roc_auc_score(positive, scores)
                metrics["auprc"] = average_precision_score(positive, scores)
    else:
        metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return metrics


def _margin_diagnostics(S: torch.Tensor, y_rows: torch.Tensor, codes: torch.Tensor) -> dict:
    """
    Semantic-geometry diagnostics from the plan's evaluation section: Frobenius
    semantic norm, mean correct-class margin, and the fraction of observations
    meeting the unit margin against every competitor.
    """
    n_classes = codes.shape[0]
    scale = (n_classes - 1) / n_classes
    scores = S @ codes.T
    true_scores = scores.gather(1, y_rows.unsqueeze(1))
    margins = scale * (true_scores - scores)
    # Exclude the true class, whose own margin is identically zero.
    margins = margins.scatter(1, y_rows.unsqueeze(1), float("inf"))
    worst = margins.min(dim=1).values
    return {
        "semantic_norm": float(torch.linalg.norm(S)),
        "mean_margin": float(worst.mean()),
        "margin_ge_1_frac": float((worst >= 1).float().mean()),
    }


def _expected_calibration_error(outcomes, probabilities, n_bins=10):
    outcomes = np.asarray(outcomes, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if mask.any():
            error += mask.mean() * abs(outcomes[mask].mean() - probabilities[mask].mean())
    return float(error)


def _calibration_metrics(y_true, probabilities, n_classes):
    from sklearn.metrics import brier_score_loss, log_loss

    y_true = np.asarray(y_true)
    if n_classes == 2:
        positive = y_true == y_true.max()
        return {
            "calibration_log_loss": log_loss(positive, probabilities, labels=[False, True]),
            "brier_score": brier_score_loss(positive, probabilities),
            "ece_10": _expected_calibration_error(positive, probabilities),
        }
    labels = np.arange(n_classes)
    one_hot = np.eye(n_classes)[y_true.astype(int)]
    classwise = [
        _expected_calibration_error(y_true == class_id, probabilities[:, class_id])
        for class_id in labels
    ]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    return {
        "calibration_log_loss": log_loss(y_true, probabilities, labels=labels),
        "brier_score": float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))),
        "ece_10": _expected_calibration_error(correct, confidence),
        "classwise_ece_10": float(np.mean(classwise)),
    }


def run_binary_config(dataset: str, strategy_name: str, seed: int,
                      pop_size=DEFAULT_POP_SIZE, n_iter=DEFAULT_N_ITER,
                      calibration: str = "none",
                      **slim_kwargs) -> dict:
    """Train and score one binary configuration on one seed."""
    from slim_gsgp.main_slim import slim

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_split(dataset, seed)
    strategy = get_strategy(strategy_name)

    started = time.time()
    model = slim(
        X_train=X_train, y_train=strategy.encode(y_train),
        X_test=X_val, y_test=strategy.encode(y_val),
        dataset_name=dataset, fitness_function=strategy.fit_string,
        pop_size=pop_size, n_iter=n_iter, seed=seed,
        log_level=0, verbose=0, test_elite=False, **slim_kwargs,
    )
    train_time = time.time() - started

    inference_started = time.time()
    raw = model.predict(X_test)
    inference_time = time.time() - inference_started

    row = {
        "dataset": dataset, "method": strategy_name, "seed": seed,
        "train_time_s": train_time, "inference_time_s": inference_time,
        "nodes_count": model.nodes_count, "n_blocks": model.size,
    }
    if slim_kwargs.get("use_adaptive_inflate"):
        row["candidate_batch"] = int(slim_kwargs.get("adaptive_candidate_batch", 1))
    row.update(evaluate_predictions(
        strategy.encode(y_test).numpy(), strategy.decode(raw).numpy(),
        scores=raw.numpy(), n_classes=2,
    ))
    if strategy_name == "margin":
        codes = simplex_codes(2)
        y_rows = (strategy.encode(y_test) > 0).long()
        row.update(_margin_diagnostics(raw.unsqueeze(1), y_rows, codes))
    if calibration != "none":
        if calibration not in ("platt", "isotonic"):
            raise ValueError("binary calibration must be 'none', 'platt', or 'isotonic'")
        calibrated = binary_probabilities(
            model.predict(X_val).numpy(), y_val.numpy(), raw.numpy(), calibration,
        )
        row["calibration_method"] = calibrated.method
        row.update(_calibration_metrics(y_test.numpy(), calibrated.probabilities, 2))
    return row


def run_multiclass_config(dataset: str, architecture: str, seed: int,
                          pop_size=DEFAULT_POP_SIZE, n_iter=DEFAULT_N_ITER,
                          objective: str = "margin", calibration: str = "none",
                          **kwargs) -> dict:
    """
    Train and score one multiclass configuration on one seed.

    ``architecture`` is ``"independent"`` (K-1 separate programs) or
    ``"shared_blocks"`` (one shared block set under the joint margin loss).

    Budgets are matched per the plan: the independent architecture evolves K-1
    programs, so each is given ``n_iter / (K-1)`` generations to keep the total
    number of individual evaluations comparable to the single shared-block run.
    """
    from slim_gsgp.classification.multiclass import fit_multiclass
    from slim_gsgp.classification.shared_blocks import fit_shared_blocks

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_split(dataset, seed)
    n_classes = len(torch.unique(y_train))

    started = time.time()
    if architecture == "independent":
        if objective != "code_regression":
            raise ValueError("independent multiclass uses code_regression only")
        per_program_iter = max(1, round(n_iter / (n_classes - 1)))
        model = fit_multiclass(X_train, y_train, X_val, y_val,
                               pop_size=pop_size, n_iter=per_program_iter,
                               log_level=0, verbose=0, seed=seed, **kwargs)
        n_blocks = sum(m.size for m in model.models)
        nodes = sum(m.nodes_count for m in model.models)
    elif architecture == "shared_blocks":
        model = fit_shared_blocks(X_train, y_train, pop_size=pop_size,
                                  n_iter=n_iter, seed=seed, objective=objective, **kwargs)
        n_blocks = model.individual.size
        nodes = model.individual.nodes_count
    else:
        raise ValueError(f"unknown architecture {architecture!r}")
    train_time = time.time() - started

    inference_started = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - inference_started

    row = {
        "dataset": dataset, "method": architecture, "seed": seed,
        "train_time_s": train_time, "inference_time_s": inference_time,
        "nodes_count": nodes, "n_blocks": n_blocks, "K": n_classes,
    }
    row.update(evaluate_predictions(y_test.numpy(), predictions.numpy(),
                                    n_classes=n_classes))
    if architecture == "shared_blocks":
        codes = model.codes
        class_to_row = {float(c): i for i, c in enumerate(model.classes)}
        y_rows = torch.tensor([class_to_row[float(c)] for c in y_test])
        row.update(_margin_diagnostics(model.semantics(X_test), y_rows, codes))
    if calibration != "none":
        if calibration != "temperature":
            raise ValueError("multiclass calibration must be 'none' or 'temperature'")
        if architecture == "independent":
            validation_semantics = torch.stack([m.predict(X_val) for m in model.models], dim=1)
            test_semantics = torch.stack([m.predict(X_test) for m in model.models], dim=1)
            validation_scores = validation_semantics @ model.codes.T
            test_scores = test_semantics @ model.codes.T
        else:
            validation_scores = model.semantics(X_val) @ model.codes.T
            test_scores = model.semantics(X_test) @ model.codes.T
        calibrated = multiclass_temperature_probabilities(
            validation_scores.numpy(), y_val.numpy(), test_scores.numpy(),
        )
        row["calibration_method"] = calibrated.method
        row.update(_calibration_metrics(y_test.numpy(), calibrated.probabilities, n_classes))
    return row


def _binary_names():
    return [spec.name for spec in BINARY_DATASETS]


def _multiclass_names():
    return [spec.name for spec in MULTICLASS_DATASETS]


def _config_hash(question, *, datasets, seeds, pop_size, n_iter, slim_version, risk, calibration):
    """Stable identifier used to prevent accidental resume across incompatible jobs."""
    payload = {
        "question": question, "datasets": datasets, "seeds": list(seeds),
        "pop_size": pop_size, "n_iter": n_iter, "slim_version": slim_version,
        "risk": risk, "calibration": calibration,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def run_question(question: str, datasets=None, seeds=DEFAULT_SEEDS,
                 pop_size=DEFAULT_POP_SIZE, n_iter=DEFAULT_N_ITER,
                 progress=True, slim_version=None, risk: str = "balanced",
                 calibration: str = "none", on_result=None, skip_keys=frozenset()) -> pd.DataFrame:
    """
    Run one experimental question over its datasets and seeds.

    Questions follow ``MS_SLIM_research_plan.md``:

    - ``q1`` sigmoid+RMSE vs raw logistic vs MS-SLIM (does removing the sigmoid help)
    - ``q2`` MS-SLIM vs code regression (does the one-sided margin matter)
    - ``q3`` the lambda grid (is the semantic regularizer useful)
    - ``q4`` independent vs shared-block multiclass (does shared structure help)
    - ``q5`` adaptive inflate vs random step across mutation steps

    Returns
    -------
    pandas.DataFrame
        Tidy results, one row per configuration and seed.
    """
    seed_range = range(seeds) if isinstance(seeds, int) else list(seeds)
    rows = []
    if risk not in ("balanced", "empirical"):
        raise ValueError("risk must be 'balanced' or 'empirical'")
    common = {"balanced": risk == "balanced"}
    if slim_version is not None:
        common["slim_version"] = slim_version
    run_hash = _config_hash(question, datasets=datasets, seeds=seed_range,
                            pop_size=pop_size, n_iter=n_iter, slim_version=slim_version,
                            risk=risk, calibration=calibration)

    def _record(fn, label, **kwargs):
        key = (question, kwargs.get("dataset"), label, kwargs.get("seed"))
        if key in skip_keys:
            return
        try:
            row = fn(**kwargs, calibration=calibration)
        except Exception as error:                  # keep a long campaign alive
            row = {"dataset": kwargs.get("dataset"), "seed": kwargs.get("seed"),
                   "method": label, "error": f"{type(error).__name__}: {error}"}
        row["method"] = label
        row["question"] = question
        row["risk"] = risk
        row["config_hash"] = run_hash
        rows.append(row)
        if on_result is not None:
            on_result(row)
        if progress:
            status = "ERROR" if "error" in row else f"{row.get('accuracy', float('nan')):.4f}"
            print(f"  {question} {str(row['dataset']):>14} {label:>18} "
                  f"seed {row['seed']:>3} -> {status}", flush=True)

    if question == "q1":
        names = datasets or _binary_names()
        for dataset, method, seed in itertools.product(
                names, ("sigmoid_rmse", "logistic", "margin"), seed_range):
            _record(run_binary_config, method, dataset=dataset, strategy_name=method,
                    seed=seed, pop_size=pop_size, n_iter=n_iter, **common)

    elif question == "q2":
        names = datasets or _binary_names()
        for dataset, method, seed in itertools.product(
                names, ("margin", "code_regression"), seed_range):
            _record(run_binary_config, method, dataset=dataset, strategy_name=method,
                    seed=seed, pop_size=pop_size, n_iter=n_iter, **common)

    elif question == "q3":
        names = datasets or _binary_names()
        for dataset, lam, seed in itertools.product(names, (0.0,) + LAMBDA_GRID, seed_range):
            _record(run_binary_config, f"margin_lam{lam:g}", dataset=dataset,
                    strategy_name="margin", seed=seed, pop_size=pop_size,
                    n_iter=n_iter, lam=lam, **common)
        for row in rows:
            row["lam"] = float(row["method"].removeprefix("margin_lam"))

    elif question == "q4":
        names = datasets or _multiclass_names()
        arms = (
            ("independent_code", "independent", "code_regression"),
            ("shared_code", "shared_blocks", "code_regression"),
            ("shared_margin", "shared_blocks", "margin"),
        )
        for dataset, arm, seed in itertools.product(names, arms, seed_range):
            label, architecture, objective = arm
            _record(run_multiclass_config, label, dataset=dataset,
                    architecture=architecture, objective=objective, seed=seed,
                    pop_size=pop_size, n_iter=n_iter, **common)

    elif question == "q5":
        names = datasets or _binary_names()
        for dataset, step, seed in itertools.product(names, MUTATION_STEPS, seed_range):
            _record(run_binary_config, f"random_ms{step:g}", dataset=dataset,
                    strategy_name="margin", seed=seed, pop_size=pop_size,
                    n_iter=n_iter, ms_lower=step, ms_upper=step, **common)
        for dataset, seed in itertools.product(names, seed_range):
            _record(run_binary_config, "adaptive", dataset=dataset,
                    strategy_name="margin", seed=seed, pop_size=pop_size,
                    n_iter=n_iter, use_adaptive_inflate=True, **common)
        for row in rows:
            operator, _, step = row["method"].partition("_ms")
            row["operator"] = operator
            row["ms"] = float(step) if step else np.nan

    elif question == "q1m":
        names = datasets or _multiclass_names()
        for dataset, objective, seed in itertools.product(names, ("margin", "cross_entropy"), seed_range):
            _record(run_multiclass_config, f"shared_{objective}", dataset=dataset,
                    architecture="shared_blocks", objective=objective, seed=seed,
                    pop_size=pop_size, n_iter=n_iter, **common)

    elif question == "q6":
        names = datasets or _binary_names()
        for dataset, candidate_batch, seed in itertools.product(names, (1, 4, 16), seed_range):
            _record(run_binary_config, f"adaptive_candidates{candidate_batch}", dataset=dataset,
                    strategy_name="margin", seed=seed, pop_size=pop_size, n_iter=n_iter,
                    use_adaptive_inflate=True, adaptive_candidate_batch=candidate_batch,
                    **common)
        for row in rows:
            row["candidate_batch"] = int(row["method"].removeprefix("adaptive_candidates"))

    elif question == "q7":
        names = datasets or _binary_names()
        for dataset, tolerance, seed in itertools.product(names, (0.0, 0.01), seed_range):
            _record(run_binary_config, f"parsimony_{tolerance:g}", dataset=dataset,
                    strategy_name="margin", seed=seed, pop_size=pop_size, n_iter=n_iter,
                    parsimony_tolerance=tolerance, **common)
        for row in rows:
            row["parsimony_tolerance"] = float(row["method"].removeprefix("parsimony_"))

    elif question == "q8":
        names = datasets or _multiclass_names()
        for dataset, geometry, seed in itertools.product(names, ("simplex", "prior_weighted"), seed_range):
            _record(run_multiclass_config, f"shared_margin_{geometry}", dataset=dataset,
                    architecture="shared_blocks", objective="margin", geometry=geometry,
                    seed=seed, pop_size=pop_size, n_iter=n_iter, **common)
        for row in rows:
            row["geometry"] = row["method"].removeprefix("shared_margin_")

    else:
        raise ValueError(f"unknown question {question!r}; expected q1, q1m, q2..q8")

    return pd.DataFrame(rows)


def paired_comparison(results: pd.DataFrame, metric: str, reference: str,
                      alpha: float = 0.05) -> pd.DataFrame:
    """
    Paired comparison of every method against a reference, per dataset.

    Uses the Wilcoxon signed-rank test on seed-matched pairs (the splits are
    reproduced from the seed, so runs are genuinely paired), Holm correction
    across the comparisons within each dataset, and the paired rank-biserial
    correlation as effect size -- the plan requires an effect size beside every
    p-value.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of ``run_question``.
    metric : str
        Column to compare, e.g. ``"balanced_accuracy"``.
    reference : str
        Method every other is compared against.
    alpha : float, optional
        Family-wise error rate (default 0.05).

    Returns
    -------
    pandas.DataFrame
        One row per (dataset, method): means, median difference, statistic,
        raw and Holm-adjusted p-values, effect size, and significance.
    """
    from scipy.stats import wilcoxon

    if "error" in results.columns:
        results = results[results["error"].isna()]

    records = []
    for dataset, block in results.groupby("dataset"):
        reference_rows = block[block["method"] == reference]
        if reference_rows.empty:
            continue
        base = reference_rows.set_index("seed")[metric]
        for method, rows in block.groupby("method"):
            if method == reference:
                continue
            other = rows.set_index("seed")[metric]
            shared = base.index.intersection(other.index)
            if len(shared) < 3:
                continue
            a, b = base.loc[shared].to_numpy(), other.loc[shared].to_numpy()
            differences = b - a
            if np.allclose(differences, 0):
                statistic, p_value, effect = 0.0, 1.0, 0.0
            else:
                statistic, p_value = wilcoxon(a, b)
                # Paired rank-biserial correlation: the signed-rank statistic
                # rescaled to [-1, 1]; positive favours `method`.
                n = int((differences != 0).sum())
                effect = 1.0 - 2.0 * statistic / (n * (n + 1) / 2)
                effect = abs(effect) if differences.mean() > 0 else -abs(effect)
            records.append({
                "dataset": dataset, "method": method, "reference": reference,
                "metric": metric, "n_pairs": len(shared),
                "reference_mean": a.mean(), "method_mean": b.mean(),
                "median_difference": float(np.median(differences)),
                "statistic": float(statistic), "p_value": float(p_value),
                "effect_size_rbc": float(effect),
            })

    table = pd.DataFrame(records)
    if table.empty:
        return table

    # Holm correction within each dataset's family of comparisons.
    adjusted = []
    for _, block in table.groupby("dataset"):
        order = block["p_value"].to_numpy().argsort()
        m = len(block)
        running, holm = 0.0, np.empty(m)
        for rank, position in enumerate(order):
            value = min(1.0, (m - rank) * block["p_value"].iloc[position])
            running = max(running, value)          # enforce monotonicity
            holm[position] = running
        part = block.copy()
        part["p_holm"] = holm
        adjusted.append(part)
    table = pd.concat(adjusted, ignore_index=True)
    table["significant"] = table["p_holm"] < alpha
    return table.sort_values(["dataset", "method"]).reset_index(drop=True)


def analyse(results: pd.DataFrame, question: str) -> pd.DataFrame:
    """Run the paired comparison appropriate to a question."""
    plans = {
        "q1": ("balanced_accuracy", "sigmoid_rmse"),
        "q2": ("balanced_accuracy", "code_regression"),
        "q3": ("balanced_accuracy", "margin_lam0"),
        "q1m": ("macro_f1", "shared_cross_entropy"),
        "q4": ("macro_f1", "shared_code"),
        "q5": ("balanced_accuracy", None),
        "q6": ("balanced_accuracy", "adaptive_candidates1"),
        "q7": ("balanced_accuracy", "parsimony_0"),
        "q8": ("macro_f1", "shared_margin_simplex"),
    }
    metric, reference = plans[question]
    if question == "q5":
        # Q5 asks about sensitivity, not level: report the spread of outcomes
        # across mutation steps for each operator.
        grouped = results.groupby(["dataset", "operator"])[metric]
        return (grouped.agg(["mean", "std", "min", "max"])
                .assign(range=lambda d: d["max"] - d["min"])
                .reset_index())
    return paired_comparison(results, metric, reference)


def _checkpoint_path(result_path: Path) -> Path:
    return result_path.with_suffix(".checkpoint.csv")


def _resume_results(result_path: Path, resume: bool) -> pd.DataFrame:
    if not resume:
        return pd.DataFrame()
    checkpoint_path = _checkpoint_path(result_path)
    source_path = checkpoint_path if checkpoint_path.exists() else result_path
    return pd.read_csv(source_path) if source_path.exists() else pd.DataFrame()


def _save_checkpoint(result_path: Path, existing: pd.DataFrame,
                     completed: list[dict]) -> Path:
    checkpoint_path = _checkpoint_path(result_path)
    pd.concat([existing, pd.DataFrame(completed)], ignore_index=True).to_csv(
        checkpoint_path, index=False,
    )
    try:
        checkpoint_path.replace(result_path)
    except PermissionError:
        return checkpoint_path
    return result_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m slim_gsgp.classification.campaign",
        description="Run the MS-SLIM experimental campaign.")
    parser.add_argument("--question", default="all",
                        choices=["all", "q1", "q1m", "q2", "q3", "q4", "q5", "q6", "q7", "q8"],
                        help="Experimental question to run (default: all).")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset names; defaults to the question's full set.")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="Number of seeds per configuration (default: 20).")
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--slim-version", default=None,
                        choices=["SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1",
                                 "SLIM+SIG2", "SLIM*SIG2"],
                        help="SLIM variant for every run (default: each fitter's own). "
                             "Q4's shared-block arm accepts only SLIM+ versions.")
    parser.add_argument("--out", type=Path, default=Path("results"),
                        help="Directory for the result CSVs (default: results/).")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress.")
    parser.add_argument("--risk", choices=["balanced", "empirical"], default="balanced")
    parser.add_argument("--calibration", default="none",
                        choices=["none", "platt", "isotonic", "temperature"])
    parser.add_argument("--resume", action="store_true",
                        help="Skip successful rows from the output CSV or its checkpoint.")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    questions = ["q1", "q1m", "q2", "q3", "q4", "q5"] if args.question == "all" else [args.question]

    for question in questions:
        print(f"== {question} ==", flush=True)
        started = time.time()
        raw_path = args.out / f"{question}.csv"
        existing = _resume_results(raw_path, args.resume)
        skip_keys = set()
        if not existing.empty:
            expected_hash = _config_hash(
                question, datasets=args.datasets, seeds=range(args.seeds),
                pop_size=args.pop_size, n_iter=args.n_iter, slim_version=args.slim_version,
                risk=args.risk, calibration=args.calibration,
            )
            same_config = existing.get("config_hash", pd.Series("", index=existing.index)) == expected_hash
            good = existing[same_config & existing.get("error", pd.Series(index=existing.index, dtype=object)).isna()]
            skip_keys = set(zip(good["question"], good["dataset"], good["method"], good["seed"]))

        completed = []
        output_path = raw_path
        reported_lock = False

        def checkpoint(row):
            nonlocal output_path, reported_lock
            completed.append(row)
            output_path = _save_checkpoint(raw_path, existing, completed)
            if output_path != raw_path and not reported_lock:
                print(f"  {raw_path} is locked; continuing in {output_path}.", flush=True)
                reported_lock = True

        results = run_question(question, datasets=args.datasets, seeds=args.seeds,
                               pop_size=args.pop_size, n_iter=args.n_iter,
                               progress=not args.quiet, slim_version=args.slim_version,
                               risk=args.risk, calibration=args.calibration,
                               on_result=checkpoint, skip_keys=skip_keys)
        if not existing.empty:
            results = pd.concat([existing, results], ignore_index=True)

        try:
            summary = analyse(results, question)
            summary.to_csv(args.out / f"{question}_analysis.csv", index=False)
            print(summary.to_string(index=False))
        except Exception as error:
            print(f"  analysis skipped: {type(error).__name__}: {error}")

        print(f"  {len(results)} runs in {time.time() - started:.1f}s -> {output_path}\n",
              flush=True)


if __name__ == "__main__":
    main()
