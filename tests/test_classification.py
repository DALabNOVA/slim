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
Correctness checks for slim_gsgp.classification (MS-SLIM margin loss and
adaptive inflate). These check the mathematical claims in
MS_SLIM_formulation.md sections 2.2 and 11, not general test coverage.
"""

import pytest
import torch
import math

from slim_gsgp.classification.codes import encode_binary, encode_zero_one, decode_binary
from slim_gsgp.classification.losses import (
    margin_loss, logistic_loss, code_regression_loss, multiclass_margin_loss,
)
from slim_gsgp.classification.adaptive_inflate import optimal_alpha
from slim_gsgp.classification.strategies import get_strategy, STRATEGIES
from slim_gsgp.classification.multiclass import simplex_codes


def test_encode_binary_maps_zero_one_to_pm1():
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    encoded = encode_binary(y)
    assert torch.equal(encoded, torch.tensor([-1.0, 1.0, 1.0, -1.0]))


def test_encode_binary_is_idempotent_on_pm1():
    y = torch.tensor([-1.0, 1.0, -1.0])
    assert torch.equal(encode_binary(y), y)


def test_decode_binary_sign_rule():
    y_pred = torch.tensor([-2.0, -0.001, 0.0, 0.001, 5.0])
    decoded = decode_binary(y_pred)
    assert torch.equal(decoded, torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0]))


def test_margin_loss_rejects_non_pm1_labels():
    loss = margin_loss(lam=0.1)
    y_true = torch.tensor([0.0, 1.0])
    y_pred = torch.tensor([0.5, -0.5])
    try:
        loss(y_true, y_pred)
        assert False, "expected ValueError for non-{-1,+1} labels"
    except ValueError:
        pass


def test_margin_loss_zero_at_lambda_zero_when_margins_satisfied():
    loss = margin_loss(lam=0.0)
    y_true = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    y_pred = torch.tensor([-2.0, 2.0, -1.0, 1.0])  # all margins exactly met or exceeded
    assert torch.isclose(loss(y_true, y_pred), torch.tensor(0.0), atol=1e-6)


def test_margin_loss_minimized_at_closed_form_optimum():
    """Formulation section 2.2: s* = y / (1 + lam), per-observation, for every lam > 0."""
    torch.manual_seed(0)
    for lam in (0.01, 0.1, 1.0, 5.0):
        loss = margin_loss(lam=lam)
        y_true = torch.tensor([1.0])
        s_star = y_true / (1.0 + lam)
        baseline = float(loss(y_true, s_star))
        for _ in range(50):
            perturbed = s_star + (torch.rand(1) - 0.5) * 2.0  # +/- 1.0 around optimum
            assert float(loss(y_true, perturbed)) >= baseline - 1e-6


def test_logistic_loss_stable_for_large_negative_margin():
    """Must not overflow to inf/nan where naive exp(-y*s) would."""
    loss = logistic_loss()
    y_true = torch.tensor([1.0])
    y_pred = torch.tensor([-1e8])
    result = loss(y_true, y_pred)
    assert torch.isfinite(result)


def test_code_regression_loss_matches_squared_error():
    loss = code_regression_loss()
    y_true = torch.tensor([-1.0, 1.0])
    y_pred = torch.tensor([0.0, 0.0])
    assert torch.isclose(loss(y_true, y_pred), torch.tensor(1.0))


def test_optimal_alpha_beats_random_alpha():
    """Formulation section 11: alpha* must not be worse than any other alpha."""
    torch.manual_seed(1)
    n, lam = 20, 0.1
    s = torch.randn(n)
    r = torch.randn(n)
    y = torch.tensor([1.0, -1.0]).repeat(n // 2)
    loss = margin_loss(lam=lam)

    alpha_star = optimal_alpha(s, r, y, lam)
    loss_star = float(loss(y, s + alpha_star * r))

    for _ in range(50):
        alpha = float((torch.rand(1) - 0.5) * 10.0)
        assert loss_star <= float(loss(y, s + alpha * r)) + 1e-5


def test_optimal_alpha_matches_dense_reference_after_active_set_changes():
    """Regression for the former three-step fixed-point approximation."""
    s = torch.tensor([-1.2, -0.4, 0.1, 0.7, 1.4, 2.1])
    r = torch.tensor([0.8, -1.3, 0.6, 1.7, -0.9, 0.4])
    y = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    loss = margin_loss(lam=0.07)
    alpha = optimal_alpha(s, r, y, 0.07)
    reference = torch.linspace(-6.0, 6.0, 24_001)
    reference_loss = torch.stack([loss(y, s + candidate * r) for candidate in reference]).min()
    assert float(loss(y, s + alpha * r)) <= float(reference_loss) + 1e-5


def test_adaptive_inflate_supports_opt_in_multiplicative_margin():
    from slim_gsgp.main_slim import slim

    X = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0], [2.5]])
    y = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    model = slim(
        X,
        y,
        fitness_function="margin",
        slim_version="SLIM*SIG1",
        use_adaptive_inflate=True,
        pop_size=8,
        n_iter=2,
        log_level=0,
        reconstruct=True,
    )
    assert math.isfinite(model.fitness)


def test_strategy_registry_has_matched_encode_decode():
    for name in STRATEGIES:
        strategy = get_strategy(name)
        assert strategy.fit_string in (
            "margin", "logistic", "code_regression", "sigmoid_rmse"
        )
        assert callable(strategy.encode)
        assert callable(strategy.decode)


def test_balanced_margin_loss_never_returns_nan_on_single_class_batch():
    """A NaN fitness is silently chosen as best by np.argmin in get_best_min."""
    loss = margin_loss(lam=0.1, balanced=True)
    y_true = torch.tensor([1.0, 1.0, 1.0])  # negative class absent
    y_pred = torch.tensor([0.5, 0.5, 0.5])
    assert torch.isfinite(loss(y_true, y_pred))


def test_encode_zero_one_maps_pm1_to_zero_one():
    """sigmoid_rmse trains on {0,1}; {-1,+1} input must be converted, not passed through."""
    assert torch.equal(encode_zero_one(torch.tensor([-1.0, 1.0, -1.0])),
                       torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(encode_zero_one(torch.tensor([0.0, 1.0])),
                       torch.tensor([0.0, 1.0]))


def test_sigmoid_rmse_strategy_encodes_to_zero_one():
    encoded = get_strategy("sigmoid_rmse").encode(torch.tensor([-1.0, 1.0]))
    assert set(encoded.tolist()) <= {0.0, 1.0}


def test_simplex_codes_is_regular_simplex():
    """Formulation section 3: unit norm, sum to zero, equal pairwise inner product."""
    for k in (2, 3, 4, 5, 8):
        codes = simplex_codes(k)
        assert codes.shape == (k, k - 1)
        assert torch.allclose(codes.norm(dim=1), torch.ones(k), atol=1e-5)
        assert torch.allclose(codes.sum(dim=0), torch.zeros(k - 1), atol=1e-5)

        gram = codes @ codes.T
        off_diag = gram[~torch.eye(k, dtype=torch.bool)]
        expected = -1.0 / (k - 1)
        assert torch.allclose(off_diag, torch.full_like(off_diag, expected), atol=1e-5)


def test_simplex_codes_k2_matches_binary_codes():
    """K=2 must reduce to the {-1,+1} codes margin_loss uses (formulation section 7)."""
    codes = simplex_codes(2)
    assert torch.equal(codes, torch.tensor([[-1.0], [1.0]]))


def test_main_slim_imports_without_priming_classification_package():
    """
    slim_config imports classification.losses, and classification's heavy modules
    import main_slim -- an eager re-export makes that a cycle that only stays
    hidden when something imports slim_gsgp.classification first. Import
    main_slim in a clean interpreter to catch a regression.
    """
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c", "from slim_gsgp.main_slim import slim"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_lazy_exports_still_reachable_from_package():
    import slim_gsgp.classification as classification
    for name in ("fit_multiclass", "predict_multiclass", "MulticlassResult", "run_question"):
        assert hasattr(classification, name)


def test_multiclass_margin_loss_optimum_at_scaled_class_code():
    """Formulation section 6.2: s_i* = c_{y_i} / (1 + lam), independent of K."""
    for k in (2, 3, 4, 5):
        codes = simplex_codes(k)
        y_rows = torch.arange(k)
        lam = 0.1
        loss = multiclass_margin_loss(codes, y_rows, lam=lam)
        optimum = codes[y_rows] / (1.0 + lam)
        baseline = float(loss(optimum))
        torch.manual_seed(0)
        for _ in range(100):
            perturbed = optimum + torch.randn(k, k - 1) * 0.3
            assert float(loss(perturbed)) >= baseline - 1e-6


def test_multiclass_margin_loss_reduces_to_binary_case():
    """Formulation section 7: K=2 must reproduce the binary margin loss exactly."""
    codes = simplex_codes(2)
    y_rows = torch.tensor([0, 1, 1, 0])
    y_binary = codes[y_rows].squeeze(1)          # {-1,+1}
    lam = 0.2
    joint = multiclass_margin_loss(codes, y_rows, lam=lam)
    binary = margin_loss(lam=lam)

    torch.manual_seed(3)
    for _ in range(20):
        s = torch.randn(4)
        assert torch.isclose(joint(s.unsqueeze(1)), binary(y_binary, s), atol=1e-6)


def test_fit_coefficients_reaches_convex_optimum():
    """A is convex given fixed block semantics; LBFGS must not improve on it."""
    from slim_gsgp.classification.shared_blocks import fit_coefficients

    torch.manual_seed(0)
    n_classes, n, n_blocks = 3, 80, 4
    codes = simplex_codes(n_classes)
    y_rows = torch.randint(0, n_classes, (n,))
    loss = multiclass_margin_loss(codes, y_rows, lam=0.05)
    R = torch.randn(n_blocks, n)

    A = fit_coefficients(R, loss, n_classes, max_iter=200)
    fitted = float(loss(R.T @ A))

    refined = A.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([refined], max_iter=200)

    def closure():
        optimizer.zero_grad()
        value = loss(R.T @ refined)
        value.backward()
        return value

    optimizer.step(closure)
    assert fitted <= float(loss(R.T @ refined.detach())) + 1e-4


def test_fit_shared_blocks_rejects_multiplicative_slim():
    from slim_gsgp.classification.shared_blocks import fit_shared_blocks
    X = torch.randn(20, 3)
    y = torch.tensor([0.0] * 7 + [1.0] * 7 + [2.0] * 6)
    try:
        fit_shared_blocks(X, y, slim_version="SLIM*ABS", pop_size=4, n_iter=1)
        assert False, "expected ValueError for multiplicative SLIM"
    except ValueError:
        pass


def test_fit_shared_blocks_learns_a_separable_problem():
    """End-to-end: shared blocks + joint loss must beat chance on clean data."""
    from slim_gsgp.classification.shared_blocks import fit_shared_blocks

    torch.manual_seed(0)
    centers = torch.tensor([[3.0, 0.0], [-3.0, 3.0], [-3.0, -3.0]])
    y_rows = torch.arange(3).repeat_interleave(25)
    X = centers[y_rows] + torch.randn(75, 2) * 0.3
    y = y_rows.float()

    result = fit_shared_blocks(X, y, pop_size=20, n_iter=8, lam=0.01, seed=0,
                               coefficient_iters=40)
    accuracy = float((result.predict(X) == y).float().mean())
    assert accuracy > 0.6, f"well-separated 3-class problem only reached {accuracy}"
    assert result.coefficients.shape == (result.individual.size, 2)


def test_paired_comparison_detects_effect_and_respects_null():
    """A known constant improvement must be significant; noise must not be."""
    import numpy as np
    import pandas as pd
    from slim_gsgp.classification.campaign import paired_comparison

    rng = np.random.default_rng(0)
    rows = []
    for seed in range(30):
        base = rng.normal(0.70, 0.03)
        rows.append({"dataset": "d", "method": "ref", "seed": seed, "balanced_accuracy": base})
        rows.append({"dataset": "d", "method": "better", "seed": seed, "balanced_accuracy": base + 0.05})
        rows.append({"dataset": "d", "method": "same", "seed": seed,
                     "balanced_accuracy": base + rng.normal(0, 0.001)})

    table = paired_comparison(pd.DataFrame(rows), "balanced_accuracy", "ref")
    better = table[table["method"] == "better"].iloc[0]
    same = table[table["method"] == "same"].iloc[0]
    assert better["significant"] and better["effect_size_rbc"] > 0.9
    assert not same["significant"]


def test_evaluate_predictions_reports_macro_f1_for_multiclass():
    from slim_gsgp.classification.campaign import evaluate_predictions

    y_true = [0, 1, 2, 0, 1, 2]
    binary = evaluate_predictions([0, 1, 0, 1], [0, 1, 0, 1], n_classes=2)
    multi = evaluate_predictions(y_true, y_true, n_classes=3)
    assert "f1" in binary and "macro_f1" not in binary
    assert "macro_f1" in multi and "f1" not in multi
    assert multi["macro_f1"] == 1.0


def test_q5_has_one_adaptive_run_per_dataset_seed():
    from slim_gsgp.classification import campaign

    original = campaign.run_binary_config
    campaign.run_binary_config = lambda **kwargs: {
        "dataset": kwargs["dataset"], "seed": kwargs["seed"], "accuracy": 0.5,
    }
    try:
        results = campaign.run_question("q5", datasets=["breast_cancer"], seeds=2,
                                        pop_size=2, n_iter=1, progress=False)
    finally:
        campaign.run_binary_config = original
    assert (results["method"] == "adaptive").sum() == 2
    assert len(results) == 12  # five random steps plus one adaptive run, twice


def test_checkpoint_survives_a_locked_result_csv(tmp_path, monkeypatch):
    import pandas as pd
    from slim_gsgp.classification import campaign

    result_path = tmp_path / "q1.csv"
    original_replace = type(result_path).replace

    def locked_replace(self, target):
        if self == campaign._checkpoint_path(result_path) and target == result_path:
            raise PermissionError("result file is open")
        return original_replace(self, target)

    monkeypatch.setattr(type(result_path), "replace", locked_replace)
    output_path = campaign._save_checkpoint(
        result_path, pd.DataFrame(), [{"dataset": "d", "seed": 0}],
    )

    assert output_path == campaign._checkpoint_path(result_path)
    resumed = campaign._resume_results(result_path, resume=True)
    assert resumed.to_dict("records") == [{"dataset": "d", "seed": 0}]


def test_q4_separates_architecture_from_objective():
    from slim_gsgp.classification import campaign

    original = campaign.run_multiclass_config
    calls = []

    def fake(**kwargs):
        calls.append((kwargs["architecture"], kwargs["objective"]))
        return {"dataset": kwargs["dataset"], "seed": kwargs["seed"], "accuracy": 0.5}

    campaign.run_multiclass_config = fake
    try:
        campaign.run_question("q4", datasets=["waveform"], seeds=1,
                              pop_size=2, n_iter=1, progress=False)
    finally:
        campaign.run_multiclass_config = original
    assert calls == [
        ("independent", "code_regression"),
        ("shared_blocks", "code_regression"),
        ("shared_blocks", "margin"),
    ]


def test_binary_calibration_uses_validation_labels_only():
    from slim_gsgp.classification.calibration import binary_probabilities

    calibrated = binary_probabilities(
        [-3.0, -1.0, 1.0, 3.0], [0, 0, 1, 1], [-2.0, 2.0], method="platt",
    )
    assert calibrated.probabilities.shape == (2,)
    assert calibrated.probabilities[0] < calibrated.probabilities[1]


def test_split_preprocessing_uses_training_median_and_columns():
    import pandas as pd
    from slim_gsgp.classification.benchmarks import _prepared_features

    train = pd.DataFrame({"x": [1.0, float("nan")], "kind": ["a", "b"]})
    test = pd.DataFrame({"x": [100.0, float("nan")], "kind": ["b", "unseen"]})
    _, columns, medians = _prepared_features(train)
    transformed, _, _ = _prepared_features(test, columns, medians)
    assert transformed[1, list(columns).index("x")] == 1.0
    assert "kind_unseen" not in columns


def test_prior_weighted_codes_are_centered_and_unit_norm():
    from slim_gsgp.classification.codes import prior_weighted_codes

    codes = prior_weighted_codes(torch.tensor([100, 20, 5]), steps=20)
    assert codes.shape == (3, 2)
    assert torch.allclose(codes.sum(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(codes.norm(dim=1), torch.ones(3), atol=1e-5)


def test_parsimony_tournament_prefers_smaller_near_tie(monkeypatch):
    import slim_gsgp.selection.selection_algorithms as selection

    class Individual:
        def __init__(self, fitness, nodes_count):
            self.fitness = fitness
            self.nodes_count = nodes_count

    class Population:
        population = [Individual(1.0, 20), Individual(1.005, 3)]

    monkeypatch.setattr(selection.random, "choices", lambda population, k: population)
    selector = selection.tournament_selection_min(2, parsimony_tolerance=0.01)
    assert selector(Population()).nodes_count == 3


SLIM_VERSIONS = ("SLIM+ABS", "SLIM*ABS", "SLIM+SIG1",
                 "SLIM*SIG1", "SLIM+SIG2", "SLIM*SIG2")
ADDITIVE_VERSIONS = tuple(v for v in SLIM_VERSIONS if v.startswith("SLIM+"))


@pytest.mark.parametrize("slim_version", SLIM_VERSIONS)
@pytest.mark.parametrize("strategy_name", ["margin", "sigmoid_rmse", "logistic",
                                           "code_regression"])
def test_binary_classification_runs_on_every_slim_variant(strategy_name, slim_version):
    """Every binary loss must train and decode under all six SLIM variants.

    Multiplicative variants are included deliberately: their semantics are a
    product of blocks, so a decoder assuming a sign change around zero could
    silently degenerate to a constant prediction.
    """
    from slim_gsgp.classification.campaign import run_binary_config

    row = run_binary_config("breast_cancer", strategy_name, seed=0,
                            pop_size=20, n_iter=10, slim_version=slim_version)
    assert "error" not in row
    assert 0.0 <= row["accuracy"] <= 1.0
    assert row["nodes_count"] > 0


@pytest.mark.parametrize("slim_version", SLIM_VERSIONS)
def test_independent_multiclass_runs_on_every_slim_variant(slim_version):
    """The K-1 independent architecture imposes no operator constraint."""
    from slim_gsgp.classification.multiclass import fit_multiclass, predict_multiclass

    torch.manual_seed(0)
    X = torch.randn(60, 4)
    y = torch.tensor([0.0, 1.0, 2.0] * 20)

    model = fit_multiclass(X, y, pop_size=20, n_iter=8, slim_version=slim_version,
                           log_level=0, verbose=0, seed=0)
    predictions = predict_multiclass(model, X)
    assert predictions.shape == y.shape
    assert set(predictions.tolist()) <= set(y.tolist())


@pytest.mark.parametrize("slim_version", ADDITIVE_VERSIONS)
def test_shared_blocks_accepts_every_additive_variant(slim_version):
    from slim_gsgp.classification.shared_blocks import fit_shared_blocks

    torch.manual_seed(0)
    X = torch.randn(45, 3)
    y = torch.tensor([0.0, 1.0, 2.0] * 15)

    model = fit_shared_blocks(X, y, pop_size=12, n_iter=4,
                              slim_version=slim_version, seed=0)
    assert model.predict(X).shape == y.shape


@pytest.mark.parametrize("slim_version", ["SLIM*ABS", "SLIM*SIG1", "SLIM*SIG2"])
def test_shared_blocks_rejects_multiplicative_variants(slim_version):
    """P(x) = sum_b r_b(x) a_b needs semantics linear in the coefficients.

    A multiplicative operator collapses blocks with prod, so no per-block
    coefficient vector can be factored out and fit_coefficients is no longer
    convex. The rejection must be explicit rather than a silently wrong fit.
    """
    from slim_gsgp.classification.shared_blocks import fit_shared_blocks

    X = torch.randn(30, 3)
    y = torch.tensor([0.0, 1.0, 2.0] * 10)

    with pytest.raises(ValueError, match="additive SLIM version"):
        fit_shared_blocks(X, y, pop_size=8, n_iter=2, slim_version=slim_version)
