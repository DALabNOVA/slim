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
This module provides various error metrics functions for evaluating machine learning models.
"""

import torch


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        RMSE value.
    """
    return torch.sqrt(torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1))


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MSE value.
    """
    return torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MAE value.
    """
    return torch.mean(torch.abs(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)


def mae_int(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE) for integer values.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MAE value for integer predictions.
    """
    return torch.mean(torch.abs(torch.sub(y_true, torch.round(y_pred))), dim=len(y_pred.shape) - 1)


def signed_errors(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute signed errors between true and predicted values.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        Signed error values.
    """
    return torch.sub(y_true, y_pred)

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute R-squared (R²) score.

    If using this fitness function, please ensure that you are maximizing the fitness value when

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        R² score value.
    """
    dim = len(y_pred.shape) - 1
    ss_res = torch.sum(torch.square(y_true - y_pred), dim=dim)
    ss_tot = torch.sum(torch.square(y_true - torch.mean(y_true)))
    # Match the useful finite convention for a constant target: a perfect
    # prediction scores 1, every other prediction scores 0.
    return torch.where(
        ss_tot == 0,
        torch.where(ss_res == 0, torch.ones_like(ss_res), torch.zeros_like(ss_res)),
        1 - (ss_res / ss_tot),
    )


def sigmoid_rmse(scaling_factor: float = 1.0):
    """
    Create a sigmoid-wrapped RMSE fitness function for binary classification.

    During training, the raw tree outputs are passed through a sigmoid to bound
    them in [0, 1] before computing RMSE against binary labels. At prediction
    time, use ``binary_sign_transform`` to convert raw outputs to class labels.

    Parameters
    ----------
    scaling_factor : float, optional
        Controls the steepness of the sigmoid (default is 1.0).

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        A fitness function with signature ``(y_true, y_pred) -> torch.Tensor``.
    """
    def _sigmoid_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        sigmoid_pred = torch.div(1, torch.add(1, torch.exp(torch.mul(-scaling_factor, y_pred))))
        return torch.sqrt(torch.mean(torch.square(torch.sub(y_true, sigmoid_pred)), dim=len(y_pred.shape) - 1))

    _sigmoid_rmse.__name__ = f"sigmoid_rmse(scaling_factor={scaling_factor})"
    return _sigmoid_rmse


def binary_sign_transform(y_pred: torch.Tensor) -> torch.Tensor:
    """
    Convert raw SLIM/GSGP outputs to binary class labels.

    All negative values are mapped to class 0, all non-negative values to class 1.
    Use this function on the output of ``Individual.predict()`` when the model was
    trained with a sigmoid-based fitness function (e.g. ``sigmoid_rmse``).

    Parameters
    ----------
    y_pred : torch.Tensor
        Raw predictions from the trained individual.

    Returns
    -------
    torch.Tensor
        Binary tensor of 0s and 1s.
    """
    return (y_pred >= 0).float()
