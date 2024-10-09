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
import pytest
import torch
from slim_gsgp.utils.utils import validate_inputs

# Dummy valid inputs to use in tests
valid_X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
valid_y_train = torch.tensor([1, 0])
valid_X_test = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
valid_y_test = torch.tensor([1, 0])
valid_pop_size = 100
valid_n_iter = 10
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
valid_tournament_size = 3

# Test for y_train type validation
def test_validate_y_train_invalid_type():
    with pytest.raises(TypeError, match="y_train must be a torch.Tensor"):
        validate_inputs(valid_X_train, "invalid", valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, valid_minimization,
                        valid_n_jobs, valid_test_elite, valid_fitness_function, valid_initializer, valid_tournament_size)

# Test for X_test type validation (optional)
def test_validate_X_test_invalid_type():
    with pytest.raises(TypeError, match="X_test must be a torch.Tensor"):
        validate_inputs(valid_X_train, valid_y_train, "invalid", valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, valid_minimization,
                        valid_n_jobs, valid_test_elite, valid_fitness_function, valid_initializer, valid_tournament_size)

# Test for y_test type validation (optional)
def test_validate_y_test_invalid_type():
    with pytest.raises(TypeError, match="y_test must be a torch.Tensor"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, "invalid", valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, valid_minimization,
                        valid_n_jobs, valid_test_elite, valid_fitness_function, valid_initializer, valid_tournament_size)

# Test for n_iter type validation
def test_validate_n_iter_invalid_type():
    with pytest.raises(TypeError, match="n_iter must be an int"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, "invalid",
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, valid_minimization,
                        valid_n_jobs, valid_test_elite, valid_fitness_function, valid_initializer, valid_tournament_size)

# Test for elitism type validation
def test_validate_elitism_invalid_type():
    with pytest.raises(TypeError, match="elitism must be a bool"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        "invalid", valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, valid_minimization,
                        valid_n_jobs, valid_test_elite, valid_fitness_function, valid_initializer, valid_tournament_size)

# Test for n_elites type validation
def test_validate_n_elites_invalid_type():
    with pytest.raises(TypeError, match="n_elites must be an int"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, "invalid", valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, valid_minimization,
                        valid_n_jobs, valid_test_elite, valid_fitness_function, valid_initializer, valid_tournament_size)

# Test for log_path type validation
def test_validate_log_path_invalid_type():
    with pytest.raises(TypeError, match="log_path must be a str"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, 123, valid_prob_const, valid_tree_functions,
                        valid_tree_constants, valid_log, valid_verbose, valid_minimization, valid_n_jobs,
                        valid_test_elite, valid_fitness_function, valid_initializer, valid_tournament_size)

# Test for tree_functions type validation
def test_validate_tree_functions_invalid_type():
    with pytest.raises(TypeError, match="tree_functions must be a non-empty list"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const, "invalid",
                        valid_tree_constants, valid_log, valid_verbose, valid_minimization, valid_n_jobs,
                        valid_test_elite, valid_fitness_function, valid_initializer, valid_tournament_size)

# Test for tree_constants type validation
def test_validate_tree_constants_invalid_type():
    with pytest.raises(TypeError, match="tree_constants must be a non-empty list"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, "invalid", valid_log, valid_verbose, valid_minimization, valid_n_jobs,
                        valid_test_elite, valid_fitness_function, valid_initializer, valid_tournament_size)

# Test for minimization type validation
def test_validate_minimization_invalid_type():
    with pytest.raises(TypeError, match="minimization must be a bool"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, "invalid", valid_n_jobs,
                        valid_test_elite, valid_fitness_function, valid_initializer, valid_tournament_size)

# Test for n_jobs type validation
def test_validate_n_jobs_invalid_type():
    with pytest.raises(TypeError, match="n_jobs must be an int"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, valid_minimization,
                        "invalid", valid_test_elite, valid_fitness_function, valid_initializer, valid_tournament_size)

# Test for fitness_function type validation
def test_validate_fitness_function_invalid_type():
    with pytest.raises(TypeError, match="fitness_function must be a str"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, valid_minimization,
                        valid_n_jobs, valid_test_elite, 123, valid_initializer, valid_tournament_size)

# Test for initializer type validation
def test_validate_initializer_invalid_type():
    with pytest.raises(TypeError, match="initializer must be a str"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, valid_minimization,
                        valid_n_jobs, valid_test_elite, valid_fitness_function, 123, valid_tournament_size)

def test_validate_tournament_size_invalid_type():
    with pytest.raises(TypeError, match="tournament_size must be an int"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, valid_minimization,
                        valid_n_jobs, valid_test_elite, valid_fitness_function, valid_initializer, "invalid")

def test_validate_tournament_size_wrong_value():
    with pytest.raises(ValueError, match="tournament_size must be at least 2"):
        validate_inputs(valid_X_train, valid_y_train, valid_X_test, valid_y_test, valid_pop_size, valid_n_iter,
                        valid_elitism, valid_n_elites, valid_init_depth, valid_log_path, valid_prob_const,
                        valid_tree_functions, valid_tree_constants, valid_log, valid_verbose, valid_minimization,
                        valid_n_jobs, valid_test_elite, valid_fitness_function, valid_initializer, -1)

