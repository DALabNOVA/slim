"""
This script sets up the configuration dictionaries for the execution of the GP algorithm
"""
from slim.algorithms.GP.operators.crossover_operators import crossover_trees
from slim.initializers.initializers import rhh, grow, full
from slim.selection.selection_algorithms import tournament_selection_min

from slim.evaluators.fitness_functions import *
from slim.utils.utils import protected_div
import torch

# Define functions and constants
# todo use only one dictionary for the parameters of each algorithm

FUNCTIONS = {
    'add': {'function': torch.add, 'arity': 2},
    'subtract': {'function': torch.sub, 'arity': 2},
    'multiply': {'function': torch.mul, 'arity': 2},
    'divide': {'function': protected_div, 'arity': 2}
}

CONSTANTS = {
    'constant_2': lambda _: torch.tensor(2.0),
    'constant_3': lambda _: torch.tensor(3.0),
    'constant_4': lambda _: torch.tensor(4.0),
    'constant_5': lambda _: torch.tensor(5.0),
    'constant__1': lambda _: torch.tensor(-1.0)
}

# Set parameters
settings_dict = {"p_test": 0.2}

# GP solve parameters
gp_solve_parameters = {
    "log": 1,
    "verbose": 1,
    "test_elite": True,
    "run_info": None,
    "ffunction": "rmse",
    "n_jobs": 1,
    "max_depth": 17,
    "n_elites":1,
    "elitism": True,
    "n_iter":1000
}

# GP parameters
gp_parameters = {
    "initializer": "rhh",
    "selector": tournament_selection_min(2),
    "crossover": crossover_trees(FUNCTIONS),
    "settings_dict": settings_dict,
    "p_xo":0.8,
    "pop_size":100,
    "seed": 74
}

gp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0,
    "init_depth": 8
}

fitness_function_options = {
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "mae_int": mae_int,
    "signed_errors": signed_errors
}

initializer_options = {
    "rhh": rhh,
    "grow": grow,
    "full": full
}