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
Shared-block multiclass MS-SLIM (integration plan phase M2;
MS_SLIM_formulation.md, section 10).

Represents the full semantic output as ``P(x) = sum_b r_b(x) * a_b``, so on the
training set ``S = R^T A`` where ``R`` is the (B, n) matrix of scalar block
semantics and ``A`` is the (B, K-1) matrix of block coefficient vectors. Each
symbolic block contributes one rank-one semantic matrix.

Unlike the independent-coordinate reference (``classification.multiclass``),
this optimizes the **joint** margin objective of formulation section 5, in which
the hinge terms couple all K-1 coordinates through ``m_ik``. One shared set of
symbolic blocks serves every class.

Implementation note: a SLIM ``Individual`` already stores exactly ``R`` in
``train_semantics``, shape (B, n) -- one scalar semantic vector per block. So no
new individual representation is needed: the blocks are evolved by the stock
``inflate``/``deflate`` operators, and only the coefficient matrix ``A`` is
added on top. ``A`` is fitted by minimizing the joint loss, which is convex in
``A`` because ``S`` is linear in it (formulation section 11).
"""

import random
import time

import numpy as np
import torch

from slim_gsgp.algorithms.GP.representations.tree import Tree as GP_Tree
from slim_gsgp.algorithms.GSGP.representations.tree import Tree
from slim_gsgp.algorithms.SLIM_GSGP.operators.mutators import (
    deflate_mutation, inflate_mutation,
)
from slim_gsgp.algorithms.SLIM_GSGP.representations.individual import Individual
from slim_gsgp.classification.codes import prior_weighted_codes, simplex_codes
from slim_gsgp.classification.losses import (
    multiclass_code_regression_loss,
    multiclass_cross_entropy_loss,
    multiclass_margin_loss,
)
from slim_gsgp.config.slim_config import FUNCTIONS, CONSTANTS, initializer_options
from slim_gsgp.utils.utils import check_slim_version, get_terminals

__all__ = ["fit_coefficients", "SharedBlockResult", "fit_shared_blocks"]


def fit_coefficients(R: torch.Tensor, loss, n_classes: int, max_iter: int = 60,
                     lr: float = 0.5) -> torch.Tensor:
    """
    Fit the block coefficient matrix ``A`` for fixed block semantics ``R``.

    ``S = R^T A`` is linear in ``A``. The supplied objectives are convex in
    ``S``, so the composed problem is convex in ``A``. LBFGS is the default
    numerical solver; convergence remains an empirical tolerance check rather
    than a claim that a fixed number of steps is exact.

    Parameters
    ----------
    R : torch.Tensor
        Block semantics, shape (B, n).
    loss : Callable
        A ``multiclass_margin_loss`` closure taking ``S`` of shape (n, K-1).
    n_classes : int
        Number of classes K.
    max_iter : int, optional
        Maximum LBFGS iterations (default 60).
    lr : float, optional
        LBFGS step size (default 0.5).

    Returns
    -------
    torch.Tensor
        Fitted coefficients ``A``, shape (B, K-1).
    """
    Rt = R.T.detach()                                    # (n, B)
    A = torch.zeros(R.shape[0], n_classes - 1, device=R.device, dtype=R.dtype,
                    requires_grad=True)
    optimizer = torch.optim.LBFGS([A], lr=lr, max_iter=max_iter,
                                  tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        optimizer.zero_grad()
        value = loss(Rt @ A)
        value.backward()
        return value

    optimizer.step(closure)
    return A.detach()


def _block_semantics(individual, X) -> torch.Tensor:
    """
    Per-block semantics ``R`` of shape (B, n) for input ``X``.

    ``Individual.predict`` collapses the blocks with sum/prod, which is exactly
    what must not happen here -- each block's scalar output is needed separately
    so it can be weighted by its own coefficient vector. This mirrors
    ``Individual.predict`` but stops before the collapse.
    """
    from slim_gsgp.algorithms.GSGP.representations.tree_utils import apply_tree

    _, sig, _ = check_slim_version(slim_version=individual.version)
    semantics = []
    for t in individual.collection:
        if isinstance(t.structure, tuple):               # base GP tree
            semantics.append(apply_tree(t, X))
        else:
            if len(t.structure) == 3:                    # one-tree mutation block
                old_train = t.structure[1].train_semantics
                t.structure[1].train_semantics = (
                    torch.sigmoid(apply_tree(t.structure[1], X)) if sig
                    else apply_tree(t.structure[1], X)
                )
                semantics.append(t.structure[0](*t.structure[1:], testing=False))
                t.structure[1].train_semantics = old_train
            elif len(t.structure) == 4:                  # two-tree mutation block
                old_train1 = t.structure[1].train_semantics
                old_train2 = t.structure[2].train_semantics
                t.structure[1].train_semantics = torch.sigmoid(apply_tree(t.structure[1], X))
                t.structure[2].train_semantics = torch.sigmoid(apply_tree(t.structure[2], X))
                semantics.append(t.structure[0](*t.structure[1:], testing=False))
                t.structure[1].train_semantics = old_train1
                t.structure[2].train_semantics = old_train2

    semantics = [s if s.numel() == len(X) else s.repeat(len(X)) for s in semantics]
    return torch.clamp(torch.stack(semantics), -1e12, 1e12)


class SharedBlockResult:
    """
    A fitted shared-block multiclass model.

    Attributes
    ----------
    individual : Individual
        The evolved SLIM individual whose blocks are shared across classes.
    coefficients : torch.Tensor
        Block coefficient matrix ``A``, shape (B, K-1).
    codes : torch.Tensor
        Simplex class codes, shape (K, K-1).
    classes : torch.Tensor
        Original class labels, in code row order.
    fitness : float
        Training loss of the returned model.
    """

    def __init__(self, individual, coefficients, codes, classes, fitness):
        self.individual = individual
        self.coefficients = coefficients
        self.codes = codes
        self.classes = classes
        self.fitness = fitness

    def semantics(self, X) -> torch.Tensor:
        """Return ``S = R^T A`` for ``X``, shape (n, K-1)."""
        return _block_semantics(self.individual, X).T @ self.coefficients

    def predict(self, X) -> torch.Tensor:
        """
        Predict class labels by ``argmax_k <s_i, c_k>`` (formulation section 3).

        Equivalent to nearest class code, since all simplex codes have unit norm.
        """
        scores = self.semantics(X) @ self.codes.T
        return self.classes[torch.argmax(scores, dim=1)]


def fit_shared_blocks(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    slim_version: str = "SLIM+ABS",
    pop_size: int = 100,
    n_iter: int = 30,
    p_inflate: float = 0.2,
    init_depth: int = 6,
    max_depth: int | None = 15,
    prob_const: float = 0.2,
    lam: float = 0.01,
    balanced: bool = False,
    coefficient_iters: int = 60,
    objective: str = "margin",
    geometry: str = "simplex",
    tournament_size: int = 2,
    seed: int = 0,
    verbose: int = 0,
) -> SharedBlockResult:
    """
    Evolve one shared set of symbolic blocks under the joint multiclass margin
    objective (formulation sections 5 and 10).

    Every individual's fitness is obtained by fitting its coefficient matrix
    ``A`` to the joint loss and reporting the resulting minimum, so selection
    ranks individuals by the best they can achieve with optimal coefficients.
    Inflate adds one shared block, deflate removes one -- the stock SLIM
    operators, unchanged.

    Parameters
    ----------
    X_train, y_train : torch.Tensor
        Training inputs and class labels (K distinct values).
    slim_version : str, optional
        SLIM variant; must be additive ("SLIM+...") for the semantic geometry
        the formulation's claims rely on (default "SLIM+ABS").
    pop_size, n_iter, p_inflate, init_depth, max_depth, prob_const : optional
        Standard SLIM search parameters.
    lam : float, optional
        Semantic regularization strength (default 0.01).
    balanced : bool, optional
        Use the class-balanced empirical risk (formulation section 8).
    coefficient_iters : int, optional
        Maximum optimizer iterations used to fit ``A`` per evaluation.
    objective : {"margin", "code_regression", "cross_entropy"}, optional
        Shared-block objective. ``code_regression`` isolates representation;
        ``cross_entropy`` is the raw-score multiclass baseline.
    geometry : {"simplex", "prior_weighted"}, optional
        Fixed simplex or exploratory class-prior-weighted code geometry.
    tournament_size : int, optional
        Tournament selection size (default 2).
    seed : int, optional
        Random seed.
    verbose : int, optional
        If non-zero, print best training loss per generation.

    Returns
    -------
    SharedBlockResult
    """
    if not slim_version.startswith("SLIM+"):
        raise ValueError(
            f"shared-block MS-SLIM requires an additive SLIM version, got "
            f"{slim_version!r}. The representation is P(x) = sum_b r_b(x) a_b, so "
            "the semantics must be linear in the coefficients for fit_coefficients "
            "to be a convex problem; a multiplicative operator collapses the blocks "
            "with prod, and no per-block coefficient vector can be factored out of "
            "a product. Use SLIM+ABS, SLIM+SIG1 or SLIM+SIG2 here. Multiplicative "
            "variants are supported by binary MS-SLIM and by the independent "
            "multiclass architecture (fit_multiclass), which impose no such "
            "constraint."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    classes = torch.unique(y_train)
    n_classes = len(classes)
    if geometry == "simplex":
        codes = simplex_codes(n_classes)
    elif geometry == "prior_weighted":
        y_counts = torch.bincount(torch.searchsorted(classes, y_train), minlength=n_classes)
        codes = prior_weighted_codes(y_counts)
    else:
        raise ValueError("geometry must be 'simplex' or 'prior_weighted'")
    codes = codes.to(device=X_train.device, dtype=X_train.dtype)
    class_to_row = {float(c): i for i, c in enumerate(classes)}
    y_rows = torch.tensor([class_to_row[float(c)] for c in y_train], device=X_train.device)
    losses = {
        "margin": lambda: multiclass_margin_loss(codes, y_rows, lam=lam, balanced=balanced),
        "code_regression": lambda: multiclass_code_regression_loss(codes, y_rows, balanced=balanced),
        "cross_entropy": lambda: multiclass_cross_entropy_loss(codes, y_rows, balanced=balanced),
    }
    try:
        loss = losses[objective]()
    except KeyError as error:
        raise ValueError("objective must be 'margin', 'code_regression', or 'cross_entropy'") from error

    operator, sig, two_trees = check_slim_version(slim_version=slim_version)

    terminals = get_terminals(X_train)
    Tree.FUNCTIONS = GP_Tree.FUNCTIONS = FUNCTIONS
    Tree.TERMINALS = GP_Tree.TERMINALS = terminals
    Tree.CONSTANTS = GP_Tree.CONSTANTS = CONSTANTS

    inflate = inflate_mutation(
        FUNCTIONS=FUNCTIONS, TERMINALS=terminals, CONSTANTS=CONSTANTS,
        two_trees=two_trees, operator=operator, sig=sig,
    )

    def evaluate(individual):
        """Fitness is the fitted coefficient objective for this block set."""
        A = fit_coefficients(individual.train_semantics, loss, n_classes,
                             max_iter=coefficient_iters)
        with torch.no_grad():
            fitness = float(loss(individual.train_semantics.T @ A))
        individual.coefficients = A
        individual.fitness = fitness
        return individual

    population = [
        Individual(
            collection=[Tree(tree, train_semantics=None, test_semantics=None,
                             reconstruct=True)],
            train_semantics=None, test_semantics=None, reconstruct=True,
        )
        for tree in initializer_options["rhh"](
            init_pop_size=pop_size, init_depth=init_depth, FUNCTIONS=FUNCTIONS,
            TERMINALS=terminals, CONSTANTS=CONSTANTS, p_c=prob_const,
        )
    ]
    for individual in population:
        individual.version = slim_version
        individual.calculate_semantics(X_train)
        evaluate(individual)

    elite = min(population, key=lambda i: i.fitness)

    for generation in range(1, n_iter + 1):
        start = time.time()
        offspring = [elite]
        while len(offspring) < pop_size:
            parent = min(random.sample(population, tournament_size),
                         key=lambda i: i.fitness)
            if random.random() < p_inflate:
                if max_depth is not None and parent.depth >= max_depth:
                    continue
                child = inflate(parent, torch.rand(1).item(), X_train,
                                max_depth=init_depth, p_c=prob_const,
                                X_test=None, reconstruct=True)
            elif parent.size > 1:
                child = deflate_mutation(parent, reconstruct=True)
            else:
                continue
            child.version = slim_version
            offspring.append(evaluate(child))

        population = offspring
        elite = min(population, key=lambda i: i.fitness)
        if verbose:
            print(f"gen {generation:3d} | loss {elite.fitness:.6f} | "
                  f"blocks {elite.size} | {time.time() - start:.2f}s")

    result = SharedBlockResult(elite, elite.coefficients, codes, classes, elite.fitness)
    result.objective = objective
    result.geometry = geometry
    return result
