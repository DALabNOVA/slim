"""Validation-only probability calibration for MS-SLIM score models."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibratedProbabilities:
    """Held-out probabilities and the method that produced them."""

    method: str
    probabilities: np.ndarray


def binary_probabilities(validation_scores, validation_y, test_scores, method="platt"):
    """Calibrate binary decision scores without exposing test labels to the fit."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    validation_scores = np.asarray(validation_scores, dtype=float).reshape(-1)
    test_scores = np.asarray(test_scores, dtype=float).reshape(-1)
    validation_y = np.asarray(validation_y).reshape(-1)
    positive = validation_y == np.max(validation_y)
    if np.unique(positive).size != 2:
        raise ValueError("binary calibration requires both classes in validation data")

    if method == "platt":
        model = LogisticRegression(C=1.0, solver="lbfgs")
        model.fit(validation_scores.reshape(-1, 1), positive)
        probabilities = model.predict_proba(test_scores.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(validation_scores, positive)
        probabilities = model.predict(test_scores)
    else:
        raise ValueError("binary calibration method must be 'platt' or 'isotonic'")
    return CalibratedProbabilities(method, np.asarray(probabilities, dtype=float))


def multiclass_temperature_probabilities(validation_scores, validation_y, test_scores):
    """Fit one temperature on validation logits and return test probabilities."""
    from scipy.optimize import minimize_scalar

    validation_scores = np.asarray(validation_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    validation_y = np.asarray(validation_y, dtype=int).reshape(-1)
    if validation_scores.ndim != 2 or test_scores.ndim != 2:
        raise ValueError("multiclass scores must have shape (n_samples, n_classes)")

    def softmax(scores, temperature):
        scaled = scores / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=1, keepdims=True)

    def objective(log_temperature):
        probabilities = softmax(validation_scores, np.exp(log_temperature))
        return -np.log(np.clip(probabilities[np.arange(len(validation_y)), validation_y], 1e-15, 1)).mean()

    fitted = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    if not fitted.success:
        raise RuntimeError("temperature calibration did not converge")
    return CalibratedProbabilities("temperature", softmax(test_scores, np.exp(fitted.x)))
