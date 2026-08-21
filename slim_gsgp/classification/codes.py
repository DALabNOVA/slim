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
Label encoding/decoding for margin-based classification (MS-SLIM).

Binary classification is the K=2 case of the simplex-code formulation
(MS_SLIM_formulation.md, section 7), so its codes are hardcoded here as
``{-1, +1}`` rather than generated from a general K-class simplex routine
that this repo does not otherwise need yet (see integration plan section 0,
anti-duplication rule 4).
"""

import torch

_POS, _NEG = 1.0, -1.0


def encode_binary(y: torch.Tensor) -> torch.Tensor:
    """
    Encode arbitrary two-valued labels as ``{-1, +1}``.

    The larger of the two distinct values in ``y`` is mapped to ``+1`` and
    the smaller to ``-1`` (so ``{0, 1}`` labels map the usual way, with
    ``1 -> +1``). Already-encoded ``{-1, +1}`` input is returned unchanged.

    Parameters
    ----------
    y : torch.Tensor
        Labels with exactly two distinct values.

    Returns
    -------
    torch.Tensor
        Float tensor with values in ``{-1.0, +1.0}``.
    """
    values = torch.unique(y)
    if values.numel() != 2:
        raise ValueError(f"encode_binary expects exactly 2 distinct labels, got {values.tolist()}")
    hi = values.max()
    return torch.where(y == hi, torch.tensor(_POS), torch.tensor(_NEG)).float()


def encode_zero_one(y: torch.Tensor) -> torch.Tensor:
    """
    Encode arbitrary two-valued labels as ``{0, 1}``, for sigmoid-based losses.

    The counterpart of ``encode_binary`` for strategies whose loss expects
    ``{0, 1}`` targets (``sigmoid_rmse``). Without this, passing ``{-1, +1}``
    labels to such a strategy trains a sigmoid output, bounded in ``[0, 1]``,
    against a ``-1`` target: the minimum is unreachable, and nothing raises.

    Parameters
    ----------
    y : torch.Tensor
        Labels with exactly two distinct values.

    Returns
    -------
    torch.Tensor
        Float tensor with values in ``{0.0, 1.0}``.
    """
    values = torch.unique(y)
    if values.numel() != 2:
        raise ValueError(f"encode_zero_one expects exactly 2 distinct labels, got {values.tolist()}")
    hi = values.max()
    return (y == hi).float()


def decode_binary(y_pred: torch.Tensor) -> torch.Tensor:
    """
    Decode raw MS-SLIM semantics to ``{-1, +1}`` class predictions.

    Prediction rule from MS_SLIM_formulation.md, section 2: ``sign(P(x))``,
    with the convention that ``P(x) == 0`` is assigned to the positive class.

    Parameters
    ----------
    y_pred : torch.Tensor
        Raw semantics from a trained individual.

    Returns
    -------
    torch.Tensor
        Tensor of ``{-1.0, +1.0}``.
    """
    return torch.where(y_pred >= 0, torch.tensor(_POS), torch.tensor(_NEG)).float()


def simplex_codes(n_classes: int) -> torch.Tensor:
    """
    Regular simplex class codes in R^(K-1) (formulation section 3).

    Built from the rows of a (K-1)-dimensional regular simplex: unit norm,
    pairwise inner product -1/(K-1), and they sum to zero. For K=2 this
    reduces to the {-1, +1} codes used by binary margin_loss (formulation
    section 7).

    Parameters
    ----------
    n_classes : int
        Number of classes K, at least 2.

    Returns
    -------
    torch.Tensor
        Shape (K, K-1). Row k is the code for class k.
    """
    if n_classes < 2:
        raise ValueError("simplex_codes requires n_classes >= 2")
    if n_classes == 2:
        return torch.tensor([[-1.0], [1.0]])

    # Standard construction: K-1 orthonormal directions of the K-point
    # regular simplex, obtained from the centered (K x K) identity via QR,
    # then normalized to unit rows.
    eye = torch.eye(n_classes)
    centered = eye - eye.mean(dim=0, keepdim=True)
    q, _ = torch.linalg.qr(centered)
    codes = q[:, : n_classes - 1]
    codes = codes / codes.norm(dim=1, keepdim=True)
    return codes


def prior_weighted_codes(class_counts: torch.Tensor, steps: int = 200,
                         lr: float = 0.05) -> torch.Tensor:
    """Learn an exploratory, prior-weighted alternative to fixed simplex codes.

    Rare-class pairs receive larger angular-separation weight. This is a
    constrained heuristic, not a theorem about optimal class geometry; callers
    must fit it from training labels only and compare it against ``simplex_codes``.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    n_classes = counts.numel()
    if n_classes < 2 or (counts <= 0).any():
        raise ValueError("class_counts must contain one positive count per class")
    codes = simplex_codes(n_classes).clone().requires_grad_(True)
    inverse = 1.0 / counts
    weights = inverse[:, None] + inverse[None, :]
    mask = ~torch.eye(n_classes, dtype=torch.bool)
    optimizer = torch.optim.Adam([codes], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        gram = codes @ codes.T
        # Low weighted cosine means stronger separation where data are rare.
        loss = (weights[mask] * gram[mask]).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            codes -= codes.mean(dim=0, keepdim=True)
            codes /= codes.norm(dim=1, keepdim=True).clamp_min(1e-8)
    # Alternating projection restores both constraints after the final Adam step.
    with torch.no_grad():
        for _ in range(100):
            codes -= codes.mean(dim=0, keepdim=True)
            codes /= codes.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return codes.detach()
