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
Fitness functions for margin-based classification (MS-SLIM) and the raw-score
baselines needed to isolate its effect (MS_SLIM_formulation.md, section 14).

All of these operate directly on raw SLIM semantics -- no sigmoid or softmax
is inserted -- and follow the same factory convention as
``slim_gsgp.evaluators.fitness_functions.sigmoid_rmse``: a hyperparameter
constructor returns a ``(y_true, y_pred) -> torch.Tensor`` callable with a
descriptive ``__name__`` for logging.
"""

import torch
import torch.nn.functional as F

_BINARY_LABELS = {-1.0, 1.0}


def _check_binary_labels(y_true: torch.Tensor) -> None:
    labels = set(torch.unique(y_true).tolist())
    if not labels <= _BINARY_LABELS:
        raise ValueError(
            f"expected labels in {{-1, +1}}, got {sorted(labels)}; "
            "use slim_gsgp.classification.codes.encode_binary(y) first"
        )


def _class_balanced_mean(per_obs: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Average per-class means instead of the per-observation mean (formulation section 8).

    Falls back to the plain mean when one class is absent: ``per_obs[empty].mean()``
    is NaN, and a NaN fitness is silently selected as the best individual by
    ``np.argmin`` in ``get_best_min``, so it must never reach the population.
    """
    pos, neg = y_true > 0, y_true < 0
    if not pos.any() or not neg.any():
        return per_obs.mean(dim=len(per_obs.shape) - 1)
    return 0.5 * (per_obs[..., pos].mean(dim=-1) + per_obs[..., neg].mean(dim=-1))


def margin_loss(lam: float = 0.01, balanced: bool = False):
    """
    Squared-hinge margin loss with semantic L2 regularization.

    ``L(s) = mean( [1 - y*s]_+^2 + lam * s^2 )``, ``y in {-1, +1}``.
    Strictly convex in ``s`` with unique optimum ``s* = y / (1 + lam)``
    (MS_SLIM_formulation.md, section 2).

    Parameters
    ----------
    lam : float, optional
        Semantic regularization strength (default 0.01). Must be positive
        for the optimum to be unique and bounded.
    balanced : bool, optional
        If True, average per-class losses instead of over all observations,
        for class-imbalanced data (formulation section 8).

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    """
    def _margin_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        _check_binary_labels(y_true)
        violation = torch.clamp(1.0 - y_true * y_pred, min=0.0) ** 2
        per_obs = violation + lam * y_pred ** 2
        return _class_balanced_mean(per_obs, y_true) if balanced else per_obs.mean(dim=len(y_pred.shape) - 1)

    _margin_loss.__name__ = f"margin_loss(lam={lam}, balanced={balanced})"
    return _margin_loss


def logistic_loss():
    """
    Raw-score logistic loss baseline: ``mean(log(1 + exp(-y*s)))``, ``y in {-1, +1}``.

    Computed via ``softplus`` for numerical stability -- the naive ``exp``
    form overflows once ``y*s`` is strongly negative, which happens routinely
    since SLIM semantics are only clamped at +/-1e12.

    Required baseline to distinguish "removing the sigmoid helps" from
    "the margin loss specifically helps" (formulation section 14, Question 1).

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    """
    def _logistic_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        _check_binary_labels(y_true)
        return F.softplus(-y_true * y_pred).mean(dim=len(y_pred.shape) - 1)

    _logistic_loss.__name__ = "logistic_loss"
    return _logistic_loss


def code_regression_loss():
    """
    Plain regression onto the class code: ``mean((y - s)^2)``.

    As a binary baseline, ``y in {-1, +1}``: isolates the effect of the
    one-sided hinge margin from the effect of using class codes and a
    distance-based prediction rule at all (formulation section 14,
    Question 2). Also reused, unmodified, as the per-coordinate loss for
    independent-coordinate multiclass MS-SLIM (``classification.multiclass``,
    integration plan phase M1), where ``y`` is a continuous simplex-code
    coordinate rather than +-1 -- squared error is well-defined either way,
    so no separate label check is applied here.

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    """
    def _code_regression_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_true - y_pred) ** 2, dim=len(y_pred.shape) - 1)

    _code_regression_loss.__name__ = "code_regression_loss"
    return _code_regression_loss


def multiclass_margin_loss(codes: torch.Tensor, y_rows: torch.Tensor, lam: float = 0.01,
                           balanced: bool = False):
    """
    Joint multiclass squared-hinge margin loss (formulation section 5).

    ``L(S) = mean_i [ 1/(K-1) * sum_{k != y_i} [1 - m_ik]_+^2 + lam * ||s_i||^2 ]``
    with the normalized margin ``m_ik = (K-1)/K * <s_i, c_{y_i} - c_k>``
    (formulation section 4). Unlike ``margin_loss``, the hinge terms couple the
    K-1 semantic coordinates through ``m_ik``; this is the objective that the
    independent-coordinate reference implementation cannot express.

    Strictly convex in ``S`` with unique optimum ``s_i* = c_{y_i} / (1 + lam)``
    (formulation section 6.2), independent of K.

    Parameters
    ----------
    codes : torch.Tensor
        Simplex class codes, shape (K, K-1), from ``codes.simplex_codes``.
    y_rows : torch.Tensor
        Row index into ``codes`` for each observation, shape (n,) of int64.
    lam : float, optional
        Semantic regularization strength (default 0.01).
    balanced : bool, optional
        If True, weight the complete per-observation loss by inverse class
        frequency (formulation section 8), which preserves both the optimum
        and strict convexity.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        ``S -> loss``, where ``S`` has shape (n, K-1) or (P, n, K-1).
    """
    n_classes = codes.shape[0]
    true_codes = codes[y_rows]                                   # (n, K-1)
    scale = (n_classes - 1) / n_classes

    # Per-observation class weights: 1/n_c normalized so a balanced dataset
    # reproduces the unweighted mean exactly.
    if balanced:
        counts = torch.bincount(y_rows, minlength=n_classes).clamp(min=1).float()
        weights = (1.0 / counts)[y_rows]
        weights = weights / weights.sum()
    else:
        weights = None

    def _multiclass_margin_loss(S: torch.Tensor) -> torch.Tensor:
        # m[i, k] = (K-1)/K * <s_i, c_y_i - c_k>, computed for all k at once.
        scores = S @ codes.T                                     # (n, K) or (P, n, K)
        true_scores = (S * true_codes).sum(dim=-1, keepdim=True)   # (n, 1) or (P, n, 1)
        margins = scale * (true_scores - scores)                  # (n, K) or (P, n, K)

        violation = torch.clamp(1.0 - margins, min=0.0) ** 2
        # Zero out the true class, which is not a competitor.
        if len(S.shape) == 3:
            violation = violation.scatter(-1, y_rows.unsqueeze(0).unsqueeze(-1).expand(S.shape[0], -1, -1), 0.0)
        else:
            violation = violation.scatter(-1, y_rows.unsqueeze(-1), 0.0)

        per_obs = violation.sum(dim=-1) / (n_classes - 1) + lam * (S ** 2).sum(dim=-1)
        if weights is not None:
            return (per_obs * weights).sum(dim=-1)
        return per_obs.mean(dim=-1)

    _multiclass_margin_loss.__name__ = (
        f"multiclass_margin_loss(K={n_classes}, lam={lam}, balanced={balanced})"
    )
    return _multiclass_margin_loss


def multiclass_code_regression_loss(codes: torch.Tensor, y_rows: torch.Tensor,
                                    balanced: bool = False):
    """Squared error to simplex codes for a complete shared-block baseline."""
    targets = codes[y_rows]
    n_classes = codes.shape[0]
    if balanced:
        counts = torch.bincount(y_rows, minlength=n_classes).clamp(min=1).float()
        weights = (1.0 / counts)[y_rows]
        weights = weights / weights.sum()
    else:
        weights = None

    def _loss(S: torch.Tensor) -> torch.Tensor:
        per_obs = (S - targets).square().sum(dim=-1)
        if weights is not None:
            return (per_obs * weights).sum(dim=-1)
        return per_obs.mean(dim=-1)

    _loss.__name__ = f"multiclass_code_regression_loss(K={n_classes}, balanced={balanced})"
    return _loss


def multiclass_cross_entropy_loss(codes: torch.Tensor, y_rows: torch.Tensor,
                                  balanced: bool = False):
    """Raw-score multiclass cross-entropy baseline on simplex class scores."""
    n_classes = codes.shape[0]
    if balanced:
        counts = torch.bincount(y_rows, minlength=n_classes).clamp(min=1).float()
        weights = (1.0 / counts)[y_rows]
        weights = weights / weights.sum()
    else:
        weights = None

    def _loss(S: torch.Tensor) -> torch.Tensor:
        scores = S @ codes.T
        per_obs = F.cross_entropy(scores, y_rows, reduction="none")
        if weights is not None:
            return (per_obs * weights).sum(dim=-1)
        return per_obs.mean(dim=-1)

    _loss.__name__ = f"multiclass_cross_entropy_loss(K={n_classes}, balanced={balanced})"
    return _loss
