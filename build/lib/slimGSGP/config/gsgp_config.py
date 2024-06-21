from slim.algorithms.GP.operators.initializers import rhh
from slim.algorithms.GP.operators.selection_algorithms import \
    tournament_selection_min
from slim.algorithms.GSGP.operators.crossover_operators import geometric_crossover
from slim.algorithms.GSGP.operators.mutators import standard_geometric_mutation
from datasets.data_loader import *
from slim.evaluators.fitness_functions import rmse
from slim.utils.utils import (generate_random_uniform, get_best_min,
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
    "reconstruct": True,
}

# GSGP parameters
gsgp_parameters = {
    "initializer": rhh,
    "selector": tournament_selection_min(2),
    "crossover": geometric_crossover,
    "mutator": standard_geometric_mutation,
    "settings_dict": settings_dict,
    "find_elit_func": get_best_min
}

gsgp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0
}

