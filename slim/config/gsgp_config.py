"""
This script sets up the configuration dictionaries for the execution of the GSGP algorithm
"""
import torch
from slim.initializers.initializers import rhh, grow, full
from slim.selection.selection_algorithms import tournament_selection_min, tournament_selection_max

from slim.algorithms.GSGP.operators.crossover_operators import geometric_crossover
from slim.algorithms.GSGP.operators.mutators import standard_geometric_mutation
from slim.evaluators.fitness_functions import *
from slim.utils.utils import (get_best_min, get_best_max,
                              protected_div)

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

# GSGP solve parameters
gsgp_solve_parameters = {
    "log": 1,
    "verbose": 1,
    "test_elite": True,
    "run_info": None,
    "ffunction": rmse,
    "reconstruct": False,
    "n_jobs": 1,
    "n_iter": 1000,
    "elitism": True,
    "n_elites": 1,
    "log": 1,
    "verbose": 1,
    "ffunction": "rmse",
    "test_elite": True
}

# GSGP parameters
gsgp_parameters = {
    "initializer": rhh,
    "selector": tournament_selection_min(2),
    "crossover": geometric_crossover,
    "mutator": standard_geometric_mutation,
    "settings_dict": settings_dict,
    "find_elit_func": get_best_min,
    "pop_size": 100,
    "p_xo": 0.0,
    "seed": 74,
    "initializer": "rhh"
}

gsgp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0.2,
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