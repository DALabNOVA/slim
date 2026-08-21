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
Benchmark datasets for the MS-SLIM experimental campaign.

Chosen against the research plan's dataset design: balanced and imbalanced
binary problems, balanced and imbalanced multiclass problems, with deliberate
variation in sample size, dimensionality and class count. Iris and wine are
deliberately excluded from the campaign registry -- they are separable enough
that every method saturates, so they cannot discriminate between the losses
under comparison. They remain useful only as smoke tests.

Datasets are fetched from OpenML on first use and cached by scikit-learn under
``~/scikit_learn_data``. Fetching requires network access; ``load_dataset``
raises a clear error rather than silently substituting anything else.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

__all__ = ["DatasetSpec", "BINARY_DATASETS", "MULTICLASS_DATASETS", "DATASETS",
           "load_dataset", "load_dataset_split", "describe_datasets"]


@dataclass(frozen=True)
class DatasetSpec:
    """
    A benchmark dataset and why it earns a place in the campaign.

    Attributes
    ----------
    name : str
        Registry key.
    loader : Callable[[], tuple]
        Returns ``(X, y)`` as float32 / int64 numpy arrays.
    task : str
        ``"binary"`` or ``"multiclass"``.
    n_classes : int
        Number of classes K.
    imbalance : str
        ``"balanced"``, ``"moderate"`` or ``"severe"``. Severe means the
        minority class is under roughly 5% of the data.
    rationale : str
        What this dataset contributes that the others do not.
    """
    name: str
    loader: Callable[[], tuple]
    task: str
    n_classes: int
    imbalance: str
    rationale: str


def _encode_targets(target):
    """Map arbitrary target labels to contiguous integer codes 0..K-1."""
    values = np.asarray(target).ravel()
    classes, codes = np.unique(values, return_inverse=True)
    return codes.astype("int64"), classes


def _openml(data_id: int):
    """Fetch an OpenML dataset by numeric id without split-dependent transforms."""
    def _load():
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        codes, _ = _encode_targets(bunch.target)
        return bunch.data.copy(), codes

    return _load


def _sklearn(name: str):
    def _load():
        from sklearn import datasets as skd

        bunch = getattr(skd, name)()
        codes, _ = _encode_targets(bunch.target)
        return bunch.data.astype("float32"), codes

    return _load


BINARY_DATASETS = (
    DatasetSpec(
        name="breast_cancer", loader=_sklearn("load_breast_cancer"),
        task="binary", n_classes=2, imbalance="moderate",
        rationale="The incumbent benchmark in this codebase, so it anchors new "
                  "results to old ones. Small and easy: a floor, not a challenge.",
    ),
    DatasetSpec(
        name="spambase", loader=_openml(44),
        task="binary", n_classes=2, imbalance="moderate",
        rationale="Many weakly informative features, so it exercises the search "
                  "rather than the loss alone.",
    ),
    DatasetSpec(
        name="phoneme", loader=_openml(1489),
        task="binary", n_classes=2, imbalance="moderate",
        rationale="Low dimensionality but a strongly nonlinear boundary, so "
                  "performance depends on the evolved structure, not feature count.",
    ),
    DatasetSpec(
        name="credit_g", loader=_openml(31),
        task="binary", n_classes=2, imbalance="moderate",
        rationale="Noisy, low-signal, and famously hard to beat the majority class "
                  "on -- separates methods that overfit from methods that do not. "
                  "Categorical columns expand under one-hot.",
    ),
    DatasetSpec(
        name="pc4", loader=_openml(1049),
        task="binary", n_classes=2, imbalance="moderate",
        rationale="Software defects with marked imbalance, where accuracy is "
                  "uninformative and AUPRC/MCC carry the signal -- exactly what "
                  "the balanced empirical risk targets.",
    ),
    DatasetSpec(
        name="churn", loader=_openml(40701),
        task="binary", n_classes=2, imbalance="moderate",
        rationale="Marked imbalance at medium scale with mixed feature types, "
                  "filling the gap between pc4 and the severe cases.",
    ),
    DatasetSpec(
        name="ozone", loader=_openml(1487),
        task="binary", n_classes=2, imbalance="severe",
        rationale="The only benchmark combining high dimensionality with severe "
                  "imbalance -- separates a loss that handles rare positives from "
                  "one that merely handles few features.",
    ),
    DatasetSpec(
        name="mammography", loader=_openml(310),
        task="binary", n_classes=2, imbalance="severe",
        rationale="Extreme imbalance at scale; the hardest test of whether the "
                  "balanced risk keeps the minority class from being ignored.",
    ),
)

MULTICLASS_DATASETS = (
    DatasetSpec(
        name="waveform", loader=_openml(60),
        task="multiclass", n_classes=3, imbalance="balanced",
        rationale="Balanced, with heavy feature noise by construction. K=3 keeps "
                  "the semantic space 2-D while still requiring a real boundary.",
    ),
    DatasetSpec(
        name="cmc", loader=_openml(23),
        task="multiclass", n_classes=3, imbalance="moderate",
        rationale="Three heavily overlapping classes with a low ceiling for every "
                  "method -- the low-signal counterpart to waveform at the same K.",
    ),
    DatasetSpec(
        name="vehicle", loader=_openml(54),
        task="multiclass", n_classes=4, imbalance="balanced",
        rationale="Small, with two heavily overlapping classes: a case where K-1 "
                  "independent programs may plausibly diverge from a shared structure.",
    ),
    DatasetSpec(
        name="page_blocks", loader=_openml(30),
        task="multiclass", n_classes=5, imbalance="severe",
        rationale="Severe multiclass imbalance at scale. Distinguishes a simplex "
                  "that fails on rare classes from one that fails on many classes; "
                  "yeast alone confounds the two.",
    ),
    DatasetSpec(
        name="satimage", loader=_openml(182),
        task="multiclass", n_classes=6, imbalance="moderate",
        rationale="Moderate K with many correlated spectral features, at a scale "
                  "where the per-individual cost of K-1 coordinates is visible.",
    ),
    DatasetSpec(
        name="segment", loader=_openml(36),
        task="multiclass", n_classes=7, imbalance="balanced",
        rationale="K high enough that the cost gap between K-1 independent programs "
                  "and one shared block set becomes measurable -- the point of Q4.",
    ),
    DatasetSpec(
        name="yeast", loader=_openml(181),
        task="multiclass", n_classes=10, imbalance="severe",
        rationale="Ten classes whose smallest holds a handful of rows. The hardest "
                  "case for a symmetric simplex and the one most likely to expose "
                  "its limits -- which the plan wants stated, not hidden.",
    ),
    DatasetSpec(
        name="pendigits", loader=_openml(32),
        task="multiclass", n_classes=10, imbalance="balanced",
        rationale="The scale case: does either architecture stay tractable when "
                  "both n and K are large?",
    ),
)

DATASETS = {spec.name: spec for spec in BINARY_DATASETS + MULTICLASS_DATASETS}


def _prepared_features(X, reference_columns=None, medians=None):
    """One-hot and median-impute features, fitting statistics only when requested."""
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        categorical = X.select_dtypes(include=["category", "object", "bool"]).columns
        frame = pd.get_dummies(X, columns=list(categorical), dummy_na=True, dtype="float32")
        if reference_columns is not None:
            frame = frame.reindex(columns=reference_columns, fill_value=0.0)
        frame = frame.astype("float32")
        values = frame.to_numpy()
        columns = frame.columns
    else:
        values = np.asarray(X, dtype="float32")
        columns = None
    if medians is None:
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
    values = np.where(np.isnan(values), medians, values).astype("float32")
    return values, columns, medians


def _raw_dataset(name: str):
    try:
        spec = DATASETS[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}")
    try:
        return spec.loader()
    except Exception as error:
        raise RuntimeError(
            f"Could not load benchmark {name!r}: {error}. OpenML datasets need "
            "network access on first use; afterwards they are cached under "
            "~/scikit_learn_data."
        ) from error


def load_dataset(name: str):
    """
    Load a registered benchmark as ``(X, y)`` torch tensors.

    Parameters
    ----------
    name : str
        A key of ``DATASETS``.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``X`` float32 of shape (n, d); ``y`` float32 class codes in 0..K-1.
    """
    X, y = _raw_dataset(name)
    X, _, _ = _prepared_features(X)

    return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)


def load_dataset_split(name: str, seed: int, p_test: float = 0.2, p_val: float = 0.2):
    """Return stratified train/validation/test tensors with train-only preprocessing."""
    X, y = _raw_dataset(name)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    def stratified_indices(indices, proportion):
        train, held_out = [], []
        for label in np.unique(y[indices]):
            members = indices[y[indices] == label].copy()
            rng.shuffle(members)
            n_holdout = max(1, int(round(len(members) * proportion))) if len(members) > 1 else 0
            held_out.extend(members[:n_holdout])
            train.extend(members[n_holdout:])
        return np.asarray(train, dtype=int), np.asarray(held_out, dtype=int)

    train_idx, test_idx = stratified_indices(np.arange(len(y)), p_test)
    train_idx, val_idx = stratified_indices(train_idx, p_val / (1.0 - p_test))
    if hasattr(X, "iloc"):
        X_train_raw, X_val_raw, X_test_raw = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    else:
        X_train_raw, X_val_raw, X_test_raw = X[train_idx], X[val_idx], X[test_idx]
    X_train, columns, medians = _prepared_features(X_train_raw)
    X_val, _, _ = _prepared_features(X_val_raw, columns, medians)
    X_test, _, _ = _prepared_features(X_test_raw, columns, medians)
    return tuple(torch.as_tensor(value, dtype=torch.float32) for value in (
        X_train, y[train_idx], X_val, y[val_idx], X_test, y[test_idx],
    ))


def describe_datasets(names=None) -> list:
    """
    Summarize registered benchmarks by actually loading them.

    Reports the real shape and class distribution rather than the values quoted
    in each ``rationale``, so the campaign's dataset table cannot drift away
    from the data it was written about.

    Parameters
    ----------
    names : iterable of str, optional
        Datasets to describe (default: all registered).

    Returns
    -------
    list of dict
        One record per dataset: name, task, n, d, K, class counts, minority share.
    """
    records = []
    for name in (names if names is not None else DATASETS):
        spec = DATASETS[name]
        X, y = load_dataset(name)
        counts = torch.bincount(y.long()).tolist()
        records.append({
            "dataset": name,
            "task": spec.task,
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "K": len(counts),
            "class_counts": counts,
            "minority_share": round(min(counts) / sum(counts), 4),
            "declared_imbalance": spec.imbalance,
        })
    return records
