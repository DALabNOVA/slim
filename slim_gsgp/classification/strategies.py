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
Classification strategies for SLIM_GSGP: each one binds together the label
encoding a fitness function expects, the fitness function itself, and the
decoding rule that turns raw semantics back into class predictions.

These three pieces must agree (e.g. margin_loss needs {-1,+1} labels and a
sign decoder; sigmoid_rmse needs {0,1} labels and a threshold-style decoder).
Picking them independently is the easiest way to get silently wrong results,
so a ``ClassificationStrategy`` binds them as one unit, and ``fit_string`` is
what you pass to ``slim_gsgp.main_slim.slim(fitness_function=...)``.
"""

from dataclasses import dataclass
from typing import Callable

import torch

from slim_gsgp.classification.codes import encode_binary, encode_zero_one, decode_binary
from slim_gsgp.evaluators.fitness_functions import binary_sign_transform


@dataclass(frozen=True)
class ClassificationStrategy:
    """
    A named, self-consistent (encoder, fitness function, decoder) triple.

    Attributes
    ----------
    name : str
        Human-readable strategy name.
    fit_string : str
        The key to pass as ``slim(fitness_function=...)``.
    encode : Callable[[torch.Tensor], torch.Tensor]
        Maps raw class labels to the target values the fitness function trains on.
    decode : Callable[[torch.Tensor], torch.Tensor]
        Maps a trained individual's raw ``predict()`` output to class predictions,
        in the same label space produced by ``encode``.
    """
    name: str
    fit_string: str
    encode: Callable[[torch.Tensor], torch.Tensor]
    decode: Callable[[torch.Tensor], torch.Tensor]


STRATEGIES = {
    "margin": ClassificationStrategy(
        name="MS-SLIM (margin loss)",
        fit_string="margin",
        encode=encode_binary,
        decode=decode_binary,
    ),
    "logistic": ClassificationStrategy(
        name="Raw-score logistic (baseline)",
        fit_string="logistic",
        encode=encode_binary,
        decode=decode_binary,
    ),
    "code_regression": ClassificationStrategy(
        name="Class-code regression (baseline)",
        fit_string="code_regression",
        encode=encode_binary,
        decode=decode_binary,
    ),
    "sigmoid_rmse": ClassificationStrategy(
        name="Sigmoid + RMSE (Bakurov et al. 2022)",
        fit_string="sigmoid_rmse",
        encode=encode_zero_one,
        decode=binary_sign_transform,
    ),
}


def get_strategy(name: str) -> ClassificationStrategy:
    """
    Look up a registered ``ClassificationStrategy`` by name.

    Parameters
    ----------
    name : str
        One of ``STRATEGIES`` keys (e.g. ``"margin"``, ``"logistic"``,
        ``"code_regression"``, ``"sigmoid_rmse"``).

    Returns
    -------
    ClassificationStrategy
    """
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(
            f"Unknown classification strategy '{name}'. Available: {sorted(STRATEGIES)}"
        )
