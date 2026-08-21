"""Exact adaptive inflate mutation for the binary squared-hinge margin loss."""

import torch

from slim_gsgp.classification.losses import margin_loss

__all__ = ["optimal_alpha", "adaptive_inflate"]


def _weights(y: torch.Tensor, balanced: bool) -> torch.Tensor:
    if not balanced:
        return torch.full_like(y, 1.0 / y.numel())
    positive, negative = y > 0, y < 0
    weights = torch.empty_like(y)
    weights[positive] = 0.5 / positive.sum().clamp(min=1)
    weights[negative] = 0.5 / negative.sum().clamp(min=1)
    return weights


def optimal_alpha(
    s: torch.Tensor,
    r: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    balanced: bool = False,
) -> torch.Tensor:
    """Return the exact minimizer of ``margin_loss(s + alpha * r)``.

    ``r`` is the affine direction.
    """
    if s.ndim != r.ndim or s.ndim != y.ndim or s.ndim != 1:
        raise ValueError("s, r, and y must be 1-dimensional tensors")
    if not (s.shape == r.shape == y.shape):
        raise ValueError("s, r, and y must have the same shape")

    y = y.to(dtype=s.dtype, device=s.device)
    r = r.to(dtype=s.dtype, device=s.device)
    weights = _weights(y, balanced)
    a = 1.0 - y * s
    b = y * r
    nonzero = b != 0

    reg_a = float(lam) * torch.sum(weights * r.square())
    reg_b = -float(lam) * torch.sum(weights * s * r)
    reg_c = float(lam) * torch.sum(weights * s.square())

    # At alpha=-infinity, b>0 hinges are active. Crossing a sorted
    # breakpoint removes a positive-b term or adds a negative-b term.
    active_left = b > 0
    q = weights * b.square()
    linear = weights * a * b
    constant = weights * a.square()
    initial_a = reg_a + q[active_left].sum()
    initial_b = reg_b + linear[active_left].sum()
    initial_c = reg_c + constant[active_left].sum()

    if not nonzero.any():
        return torch.zeros((), dtype=s.dtype, device=s.device)

    breakpoints, order = torch.sort(a[nonzero] / b[nonzero])
    event_b = b[nonzero][order]
    sign = torch.where(event_b > 0, -torch.ones_like(event_b), torch.ones_like(event_b))
    event_q = q[nonzero][order]
    event_linear = linear[nonzero][order]
    event_constant = constant[nonzero][order]

    coefficients_a = torch.cat((initial_a.unsqueeze(0), initial_a.unsqueeze(0) + torch.cumsum(sign * event_q, 0)))
    coefficients_b = torch.cat((initial_b.unsqueeze(0), initial_b.unsqueeze(0) + torch.cumsum(sign * event_linear, 0)))
    coefficients_c = torch.cat((initial_c.unsqueeze(0), initial_c.unsqueeze(0) + torch.cumsum(sign * event_constant, 0)))

    roots = torch.where(coefficients_a > 0, coefficients_b / coefficients_a, torch.zeros_like(coefficients_a))
    lower = torch.cat((torch.full_like(breakpoints[:1], -torch.inf), breakpoints))
    upper = torch.cat((breakpoints, torch.full_like(breakpoints[:1], torch.inf)))
    candidates = torch.minimum(torch.maximum(roots, lower), upper)
    candidates = torch.cat((candidates, torch.zeros(1, dtype=s.dtype, device=s.device)))

    # Each interval is an exact quadratic, including its endpoints. Evaluating
    # all interval minimizers handles repeated breakpoints and lambda=0.
    losses = coefficients_a * candidates[:-1].square() - 2 * coefficients_b * candidates[:-1] + coefficients_c
    zero_loss = torch.sum(weights * torch.clamp(a, min=0).square()) + reg_c
    losses = torch.cat((losses, zero_loss.unsqueeze(0)))
    minimum = losses.min()
    tied_distance = torch.where(losses == minimum, candidates.abs(), torch.full_like(candidates, torch.inf))
    return candidates[torch.argmin(tied_distance)]


def adaptive_inflate(base_inflate, y_train: torch.Tensor, lam: float, operator: str = "sum",
                     balanced: bool = False, candidate_batch: int = 1):
    """Wrap an inflate mutator with an exact margin-loss line search.

    For addition the search direction is the raw block ``r``. For product
    variants, ``s * (1 + alpha*r) = s + alpha*(s*r)``, so the same convex
    solver applies with direction ``s*r``. Random-step SLIM* is unchanged
    unless this opt-in wrapper is requested.
    """
    if operator not in ("sum", "mul"):
        raise ValueError("adaptive_inflate operator must be 'sum' or 'mul'")
    if candidate_batch < 1:
        raise ValueError("adaptive inflate candidate_batch must be at least 1")

    def _adaptive_inflate(individual, ms, X, **kwargs):
        collapse = torch.sum if operator == "sum" else torch.prod
        s = collapse(individual.train_semantics, dim=0)
        candidates = []
        objective = margin_loss(lam=lam, balanced=balanced)
        for _ in range(candidate_batch):
            offspring = base_inflate(individual, 1.0, X, **kwargs)
            unit_block = offspring.train_semantics[-1]
            if operator == "sum":
                raw_block = unit_block
                direction = raw_block
            else:
                raw_block = unit_block - 1.0
                direction = s * raw_block
            alpha = optimal_alpha(s, direction, y_train, lam, balanced=balanced)
            candidates.append((float(objective(y_train, s + alpha * direction)), offspring, raw_block, alpha))
        _, offspring, raw_block, alpha = min(candidates, key=lambda candidate: candidate[0])
        if operator == "sum":
            offspring.train_semantics[-1] = raw_block * alpha
            if offspring.test_semantics is not None:
                offspring.test_semantics[-1] = offspring.test_semantics[-1] * alpha
        else:
            offspring.train_semantics[-1] = 1.0 + alpha * raw_block
            if offspring.test_semantics is not None:
                offspring.test_semantics[-1] = 1.0 + alpha * (offspring.test_semantics[-1] - 1.0)

        if hasattr(offspring, "collection"):
            offspring.collection[-1].structure[-1] = float(alpha)
        offspring.adaptive_candidate_batch = candidate_batch
        return offspring

    return _adaptive_inflate
