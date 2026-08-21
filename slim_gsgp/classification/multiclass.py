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
Independent-coordinate multiclass MS-SLIM (integration plan, phase M1;
MS_SLIM_formulation.md, section 10, reference implementation).

Evolves K-1 independent SLIM programs, one per simplex coordinate, each
trained with ``code_regression_loss`` against that coordinate of the true
class's code. Prediction picks the class whose code is closest to the
stacked output.

Each coordinate's target is a continuous simplex-code value (e.g. 0.5 or
-0.866, not +-1), so the binary squared-hinge ``margin_loss`` does not apply
per coordinate -- formulation section 5's margin ``m_ik`` is a joint quantity
across all K-1 coordinates together, not something with a per-coordinate
hinge form. ``code_regression_loss`` is the formulation-sanctioned baseline
(section 14, Question 2) that regresses each coordinate directly onto its
code value instead.

This means M1 is a plain-regression approximation of MS-SLIM's prediction
rule and class geometry, not its margin objective -- the margin/hinge
structure only exists once coordinates are coupled, which requires the
shared-block representation of phase M2. State this when reporting results
(see integration plan, section 4, phase M1).

No core SLIM files are touched -- this runs plain ``slim()`` K-1 times.
"""

import torch

from slim_gsgp.main_slim import slim
from slim_gsgp.classification.codes import simplex_codes

__all__ = ["MulticlassResult", "fit_multiclass", "predict_multiclass"]




class MulticlassResult:
    """
    Holds the K-1 fitted per-coordinate models and the class codes used.

    Attributes
    ----------
    models : list
        One fitted ``Individual`` per simplex coordinate (length K-1).
    codes : torch.Tensor
        Shape (K, K-1), from ``simplex_codes``.
    classes : torch.Tensor
        Shape (K,), the original class labels in code row order.
    """

    def __init__(self, models: list, codes: torch.Tensor, classes: torch.Tensor):
        self.models = models
        self.codes = codes
        self.classes = classes

    def predict(self, X) -> torch.Tensor:
        """Predict class labels for ``X`` (see ``predict_multiclass``)."""
        return predict_multiclass(self, X)


def fit_multiclass(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor = None,
    y_val: torch.Tensor = None,
    **slim_kwargs,
) -> MulticlassResult:
    """
    Fit K-1 independent ``code_regression_loss`` SLIM models, one per
    simplex coordinate.

    Parameters
    ----------
    X_train : torch.Tensor
        Training inputs.
    y_train : torch.Tensor
        Training class labels (any dtype with K distinct values).
    X_val, y_val : torch.Tensor, optional
        Validation set forwarded to ``slim()`` as its internal ``X_test``/
        ``y_test`` (never the held-out test set).
    **slim_kwargs
        Forwarded to ``slim_gsgp.main_slim.slim`` for every coordinate run
        (e.g. ``pop_size``, ``n_iter``, ``slim_version``).

    Returns
    -------
    MulticlassResult
    """
    classes = torch.unique(y_train)
    codes = simplex_codes(len(classes))

    class_to_row = {float(c): i for i, c in enumerate(classes)}
    row_idx = torch.tensor([class_to_row[float(c)] for c in y_train])
    y_codes = codes[row_idx]  # (n, K-1)

    val_codes = None
    if y_val is not None:
        unseen = {float(c) for c in torch.unique(y_val)} - set(class_to_row)
        if unseen:
            raise ValueError(
                f"y_val contains classes absent from y_train: {sorted(unseen)}; "
                "they have no simplex code, so no validation target can be built"
            )
        val_row_idx = torch.tensor([class_to_row[float(c)] for c in y_val])
        val_codes = codes[val_row_idx]

    models = []
    for j in range(codes.shape[1]):
        target_train = y_codes[:, j]
        target_val = val_codes[:, j] if val_codes is not None else None
        model = slim(
            X_train=X_train, y_train=target_train,
            X_test=X_val, y_test=target_val,
            fitness_function="code_regression",
            **slim_kwargs,
        )
        models.append(model)

    return MulticlassResult(models, codes, classes)


def predict_multiclass(result: MulticlassResult, X) -> torch.Tensor:
    """
    Predict class labels: argmax_k <stacked output, code_k> (formulation
    section 3), equivalent to nearest class code by Euclidean distance since
    all codes have equal norm.

    Parameters
    ----------
    result : MulticlassResult
    X : torch.Tensor
        Inputs to predict on.

    Returns
    -------
    torch.Tensor
        Predicted class labels, same dtype/values as the original training labels.
    """
    outputs = torch.stack([m.predict(X) for m in result.models], dim=1)  # (n, K-1)
    scores = outputs @ result.codes.T  # (n, K)
    best = torch.argmax(scores, dim=1)
    return result.classes[best]
