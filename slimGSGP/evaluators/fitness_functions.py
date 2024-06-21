"""
This module provides various error metrics functions for evaluating machine learning models.
"""

import torch


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE).

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: RMSE value.
    """
    return torch.sqrt(
        torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)
    )


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE).

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: MSE value.
    """
    return torch.mean(
        torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1
    )


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE).

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: MAE value.
    """
    return torch.mean(torch.abs(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)


def mae_int(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE) for integer values.

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: MAE value for integer predictions.
    """
    return torch.mean(
        torch.abs(torch.sub(y_true, torch.round(y_pred))), dim=len(y_pred.shape) - 1
    )


def signed_errors(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute signed errors between true and predicted values.

    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: Signed error values.
    """
    return torch.sub(y_true, y_pred)
