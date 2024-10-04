import torch
from slim.initializers.initializers import rhh, grow, full
from slim.algorithms.GSGP.operators.crossover_operators import geometric_crossover
from slim.algorithms.SLIM_GSGP.operators.mutators import (deflate_mutation)
from slim.algorithms.SLIM_GSGP.operators.selection_algorithms import \
    tournament_selection_min_slim
from slim.evaluators.fitness_functions import *
from slim.utils.utils import (get_best_min, protected_div)

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

# SLIM GSGP solve parameters
slim_gsgp_solve_parameters = {
    "log": 1,
    "verbose": 1,
    "run_info": None,
    "ffunction": rmse,
    "max_depth": 5,
    "reconstruct": True
}

# SLIM GSGP parameters
slim_gsgp_parameters = {
    "initializer": full,
    "selector": tournament_selection_min_slim(2),
    "crossover": geometric_crossover,
    "ms": None,
    "inflate_mutator": None,
    "deflate_mutator": deflate_mutation,
    "p_xo": 0,
    "settings_dict": settings_dict,
    "find_elit_func": get_best_min,
    "p_inflate": None,
    "copy_parent": None,
    "operator": None
}
slim_gsgp_parameters["p_m"] = 1 - slim_gsgp_parameters["p_xo"]

slim_gsgp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 1
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