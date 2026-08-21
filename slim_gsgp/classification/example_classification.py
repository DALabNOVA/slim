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
Runnable tour of ``slim_gsgp.classification``.

Run it directly::

    python -m slim_gsgp.classification.example_classification

Covers, in order: choosing a technique with a strategy, adaptive inflate,
both multiclass architectures, and the experiment harness. Budgets are kept
small so the whole script finishes quickly; raise ``POP_SIZE`` and ``N_ITER``
for real runs.
"""

import torch

from slim_gsgp.classification import (
    fit_multiclass, fit_shared_blocks, get_strategy,
)
from slim_gsgp.datasets.data_loader import load_breast_cancer
from slim_gsgp.main_slim import slim
from slim_gsgp.utils.utils import train_test_split

POP_SIZE, N_ITER = 50, 20


def _breast_cancer_splits():
    X, y = load_breast_cancer(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=0)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _iris_splits():
    from sklearn.datasets import load_iris

    data = load_iris()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32)
    return train_test_split(X, y, p_test=0.3, seed=0)


def binary_margin():
    """MS-SLIM on a binary problem, via the strategy registry."""
    X_train, X_val, X_test, y_train, y_val, y_test = _breast_cancer_splits()

    strategy = get_strategy("margin")
    model = slim(
        X_train=X_train, y_train=strategy.encode(y_train),
        X_test=X_val, y_test=strategy.encode(y_val),
        dataset_name="breast_cancer", slim_version="SLIM+ABS",
        pop_size=POP_SIZE, n_iter=N_ITER,
        fitness_function=strategy.fit_string, lam=0.01,
        log_level=0, verbose=0, seed=0,
    )
    predictions = strategy.decode(model.predict(X_test))
    accuracy = float((predictions == strategy.encode(y_test)).float().mean())
    print(f"[margin]           accuracy {accuracy:.4f}  blocks {model.size}")


def binary_adaptive_inflate():
    """The same objective, with the inflate step solved instead of sampled."""
    X_train, X_val, X_test, y_train, y_val, y_test = _breast_cancer_splits()

    strategy = get_strategy("margin")
    model = slim(
        X_train=X_train, y_train=strategy.encode(y_train),
        X_test=X_val, y_test=strategy.encode(y_val),
        dataset_name="breast_cancer", slim_version="SLIM+ABS",
        pop_size=POP_SIZE, n_iter=N_ITER,
        fitness_function="margin", lam=0.01, use_adaptive_inflate=True,
        log_level=0, verbose=0, seed=0,
    )
    predictions = strategy.decode(model.predict(X_test))
    accuracy = float((predictions == strategy.encode(y_test)).float().mean())
    print(f"[adaptive inflate] accuracy {accuracy:.4f}  blocks {model.size}")


def multiclass_independent():
    """K-1 separate programs, one per simplex coordinate."""
    X_train, X_test, y_train, y_test = _iris_splits()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, p_test=0.25, seed=0)

    model = fit_multiclass(X_train, y_train, X_val, y_val,
                           slim_version="SLIM+ABS", pop_size=POP_SIZE, n_iter=N_ITER,
                           log_level=0, verbose=0, seed=0)
    accuracy = float((model.predict(X_test) == y_test).float().mean())
    print(f"[independent]      accuracy {accuracy:.4f}  "
          f"blocks {sum(m.size for m in model.models)} across {len(model.models)} programs")


def multiclass_shared_blocks():
    """One shared block set for every class, under the joint margin objective."""
    X_train, X_test, y_train, y_test = _iris_splits()

    model = fit_shared_blocks(X_train, y_train, slim_version="SLIM+ABS",
                              pop_size=POP_SIZE, n_iter=N_ITER, lam=0.01, seed=0)
    accuracy = float((model.predict(X_test) == y_test).float().mean())
    print(f"[shared blocks]    accuracy {accuracy:.4f}  blocks {model.individual.size} "
          f"(one shared set), coefficients {tuple(model.coefficients.shape)}")


def compare_strategies():
    """Paired comparison across strategies on identical stratified splits."""
    import pandas as pd
    from slim_gsgp.classification.campaign import run_binary_config

    for name in ("margin", "logistic", "code_regression", "sigmoid_rmse"):
        results = pd.DataFrame([
            run_binary_config("breast_cancer", name, seed=seed,
                              pop_size=POP_SIZE, n_iter=N_ITER,
                              slim_version="SLIM+ABS")
            for seed in range(3)
        ])
        print(f"[{name:>15}]  accuracy {results.accuracy.mean():.4f}"
              f"  auroc {results.auroc.mean():.4f}"
              f"  nodes {results.nodes_count.mean():.0f}")


if __name__ == "__main__":
    print("== binary ==")
    binary_margin()
    binary_adaptive_inflate()
    print("\n== multiclass (iris) ==")
    multiclass_independent()
    multiclass_shared_blocks()
    print("\n== strategy comparison, 3 seeds, identical splits ==")
    compare_strategies()
