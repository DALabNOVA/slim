from slim.utils.utils import validate_functions_dictionary
import torch

FUNCTIONS = {
    'add': {'function': "huh", 'arity': 2},
    'subtract': {'function': torch.sub, 'arity': 2},
    'multiply': {'function': torch.mul, 'arity': 2}
}

validate_functions_dictionary(FUNCTIONS)