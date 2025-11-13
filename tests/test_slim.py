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
from slim_gsgp.main_gp import gp  # import the slim_gsgp library
import pytest
import torch
from slim_gsgp.main_slim import slim  # import the slim_gsgp library
from slim_gsgp.datasets.data_loader import load_ppb  # import the loader for the dataset PPB
from slim_gsgp.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim_gsgp.utils.utils import train_test_split  # import the train-test split function

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

valid_result = 53.71623611450195

def test_slim_valid_inputs():
    X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_train = torch.tensor([1, 0])
    # Call slim_gsgp with valid inputs. Number of generations is lowered for running time purposes
    result = slim(X_train, y_train, n_iter=3)
    assert result is not None  # Check if function returns valid output

def test_slim_invalid_X_train():
    y_train = torch.tensor([1, 0])
    with pytest.raises(TypeError, match="X_train must be a torch.Tensor"):
        slim("invalid_type", y_train)

# Test for invalid pop_size (should be int)
def test_slim_invalid_pop_size():
    X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_train = torch.tensor([1, 0])
    with pytest.raises(TypeError, match="pop_size must be an int"):
        slim(X_train, y_train, pop_size="invalid_type")

# Test for invalid prob_const (should be float)
def test_slim_invalid_prob_const():
    X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_train = torch.tensor([1, 0])
    with pytest.raises(TypeError, match="prob_const must be a float"):
        slim(X_train, y_train, prob_const="invalid_type")

# Test for out-of-range prob_const (should be between 0 and 1)
def test_slim_out_of_range_prob_const():
    X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_train = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="prob_const must be a number between 0 and 1"):
        slim(X_train, y_train, prob_const=1.5)

def test_slim_min_n_iter():
    with pytest.raises(ValueError, match="n_iter must be greater than 0"):
        slim(valid_X_train, valid_y_train, n_iter=0)  # n_iter too small

def test_slim_seed_reproducibility():
    result1 = slim(valid_X_train, valid_y_train, seed=42, n_iter=valid_n_iter, reconstruct=True)
    result2 = slim(valid_X_train, valid_y_train, seed=42, n_iter=valid_n_iter, reconstruct=True)
    assert torch.equal(result1.predict(valid_X_test), result2.predict(valid_X_test)), \
            "Results should be reproducible with the same seed"

def test_slim_n_jobs_parallelism():
    result = slim(valid_X_train, valid_y_train, n_jobs=4, n_iter=valid_n_iter)
    assert result is not None, "Function should run successfully in parallel"

def test_slim_immutability():
    X, y = load_ppb(X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

    final_tree = slim(X_train=X_train, y_train=y_train,
                      X_test=X_val, y_test=y_val,
                      slim_version='SLIM+SIG2', pop_size=100, n_iter=100,
                      ms_lower=0, ms_upper=1, p_inflate=0.5, reconstruct=True, init_depth=8, copy_parent=None)

    predictions = final_tree.predict(X_test)
    print(float(rmse(y_true=y_test, y_pred=predictions)))
    assert float(rmse(y_true=y_test, y_pred=predictions)) == valid_result, "Final result should not change with updates"

def test_slim_saving_and_loading():
    result_tree = slim(valid_X_train, valid_y_train, n_iter=valid_n_iter, reconstruct=True)
    result_tree.save_to_file('test_slim_saving.txt')
    loaded_tree = type(result_tree).load_from_file('test_slim_saving.txt')
    assert torch.equal(result_tree.predict(valid_X_test), loaded_tree.predict(valid_X_test)), \
        "Loaded tree should produce the same predictions as the original"