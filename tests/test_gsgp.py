from slim.main_gp import gp  # import the slim library
import pytest
import torch
from slim.main_gsgp import gsgp  # import the slim library
from slim.datasets.data_loader import load_ppb  # import the loader for the dataset PPB
from slim.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim.utils.utils import train_test_split  # import the train-test split function
# NOTE: The number of generations is lowered in most tests to prevent unnecessary running times when testing.


# Dummy valid inputs to use in tests
valid_X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
valid_y_train = torch.tensor([1, 0])
valid_X_test = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
valid_y_test = torch.tensor([1, 0])
valid_pop_size = 30
valid_n_iter = 3
valid_elitism = True
valid_n_elites = 2
valid_init_depth = 3
valid_log_path = "log_path.csv"
valid_prob_const = 0.5
valid_tree_functions = ["add", "sub"]
valid_tree_constants = [1.0, 2.0]
valid_log = 2
valid_verbose = 1
valid_minimization = True
valid_n_jobs = 1
valid_test_elite = False
valid_fitness_function = "mean_squared_error"
valid_initializer = "random"

valid_result = 57.38235092163086


def test_gsgp_valid_inputs():
    X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_train = torch.tensor([1, 0])
    # Call gsgp with valid inputs. Number of generations is lowered for running time purposes
    result = gsgp(X_train, y_train, n_iter=3)
    assert result is not None  # Check if function returns valid output

def test_gsgp_invalid_X_train():
    y_train = torch.tensor([1, 0])
    with pytest.raises(TypeError, match="X_train must be a torch.Tensor"):
        gsgp("invalid_type", y_train)

# Test for invalid pop_size (should be int)
def test_gsgp_invalid_pop_size():
    X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_train = torch.tensor([1, 0])
    with pytest.raises(TypeError, match="pop_size must be an int"):
        gsgp(X_train, y_train, pop_size="invalid_type")

# Test for invalid prob_const (should be float)
def test_gsgp_invalid_prob_const():
    X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_train = torch.tensor([1, 0])
    with pytest.raises(TypeError, match="prob_const must be a float"):
        gsgp(X_train, y_train, prob_const="invalid_type")

# Test for out-of-range prob_const (should be between 0 and 1)
def test_gsgp_out_of_range_prob_const():
    X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_train = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="prob_const must be a number between 0 and 1"):
        gsgp(X_train, y_train, prob_const=1.5)

def test_gsgp_min_n_iter():
    with pytest.raises(ValueError, match="n_iter must be greater than 0"):
        gsgp(valid_X_train, valid_y_train, n_iter=0)  # n_iter too small

def test_gsgp_seed_reproducibility():
    result1 = gsgp(valid_X_train, valid_y_train, seed=42, n_iter=valid_n_iter, reconstruct=True)
    result2 = gsgp(valid_X_train, valid_y_train, seed=42, n_iter=valid_n_iter, reconstruct=True)
    assert torch.equal(result1.predict(valid_X_test), result2.predict(valid_X_test)), \
            "Results should be reproducible with the same seed"

def test_gsgp_n_jobs_parallelism():
    result1 = gsgp(valid_X_train, valid_y_train, n_jobs=4, n_iter=valid_n_iter, reconstruct=True)
    assert result1 is not None, "Function should run successfully in parallel"
    result2 = gsgp(valid_X_train, valid_y_train, n_jobs=1, n_iter=valid_n_iter, reconstruct=True)
    assert torch.equal(result1.predict(valid_X_test), result2.predict(valid_X_test)), \
        "Results should be the same with the same seed and different n_jobs"


def test_gsgp_immutability():
    X, y = load_ppb(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

    final_tree = gsgp(X_train=X_train, y_train=y_train,
                      X_test=X_val, y_test=y_val, p_xo=0,
                      dataset_name='ppb', pop_size=100, n_iter=10,
                      reconstruct=True, ms_lower=0, ms_upper=1)

    predictions = final_tree.predict(X_test)
    assert float(rmse(y_true=y_test, y_pred=predictions)) == valid_result, "Final result should not change with updates"
