from slim.algorithms.GP.operators.crossover_operators import crossover_trees
from slim.algorithms.GP.operators.initializers import rhh
from slim.algorithms.GP.operators.selection_algorithms import \
    tournament_selection_min

from datasets.data_loader import *
from slim.evaluators.fitness_functions import rmse
from slim.utils.utils import (get_best_max, get_best_min,
                              protected_div)

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
    "max_": False,
    "ffunction": rmse,
    "tree_pruner": None
}

# GP parameters
gp_parameters = {
    "initializer": rhh,
    "selector": tournament_selection_min(2),
    "crossover": crossover_trees(FUNCTIONS),
    "settings_dict": settings_dict,
    "find_elit_func": get_best_max if gp_solve_parameters["max_"] else get_best_min
}

gp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0
}
