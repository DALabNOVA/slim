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
This script runs the SLIM_GSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import uuid
import os
import warnings

from slim_gsgp.algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from slim_gsgp.config.slim_config import *
from slim_gsgp.utils.logger import log_settings
from slim_gsgp.utils.utils import (get_terminals, check_slim_version, validate_inputs, generate_random_uniform,
                                   get_best_min, get_best_max)
from slim_gsgp.algorithms.SLIM_GSGP.operators.mutators import inflate_mutation
from slim_gsgp.selection.selection_algorithms import tournament_selection_max, tournament_selection_min


ELITES = {}
UNIQUE_RUN_ID = uuid.uuid1()


def slim(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
         dataset_name: str = None,
         slim_version: str = "SLIM+SIG2",
         pop_size: int = slim_gsgp_parameters["pop_size"],
         n_iter: int = slim_gsgp_solve_parameters["n_iter"],
         elitism: bool = slim_gsgp_solve_parameters["elitism"], n_elites: int = slim_gsgp_solve_parameters["n_elites"],
         init_depth: int = slim_gsgp_pi_init["init_depth"],
         ms_lower: float = 0, ms_upper: float = 1,
         p_inflate: float = slim_gsgp_parameters["p_inflate"],
         log_path: str = None,
         seed: int = slim_gsgp_parameters["seed"],
         log_level: int = slim_gsgp_solve_parameters["log"],
         verbose: int = slim_gsgp_solve_parameters["verbose"],
         reconstruct: bool = slim_gsgp_solve_parameters["reconstruct"],
         fitness_function: str = slim_gsgp_solve_parameters["ffunction"],
         initializer: str = slim_gsgp_parameters["initializer"],
         minimization: bool = True,
         prob_const: float = slim_gsgp_pi_init["p_c"],
         tree_functions: list = list(FUNCTIONS.keys()),
         tree_constants: list = [float(key.replace("constant_", "").replace("_", "-")) for key in CONSTANTS],
         copy_parent: bool =slim_gsgp_parameters["copy_parent"],
         max_depth: int | None = slim_gsgp_solve_parameters["max_depth"],
         n_jobs: int = slim_gsgp_solve_parameters["n_jobs"],
         tournament_size: int = 2,
         test_elite: bool = slim_gsgp_solve_parameters["test_elite"]):

    """
    Main function to execute the SLIM GSGP algorithm on specified datasets.

    Parameters
    ----------
    X_train: (torch.Tensor)
        Training input data.
    y_train: (torch.Tensor)
        Training output data.
    X_test: (torch.Tensor), optional
        Testing input data.
    y_test: (torch.Tensor), optional
        Testing output data.
    dataset_name : str, optional
        Dataset name, for logging purposes
    pop_size : int, optional
        The population size for the genetic programming algorithm (default is 100).
    n_iter : int, optional
        The number of iterations for the genetic programming algorithm (default is 100).
    elitism : bool, optional
        Indicate the presence or absence of elitism.
    n_elites : int, optional
        The number of elites.
    init_depth : int, optional
        The depth value for the initial GP trees population.
    ms_lower : float, optional
        Lower bound for mutation rates (default is 0).
    ms_upper : float, optional
        Upper bound for mutation rates (default is 1).
    p_inflate : float, optional
        Probability of selecting inflate mutation when mutating an individual.
    log_path : str, optional
        The path where is created the log directory where results are saved.
    seed : int, optional
        Seed for the randomness
    log_level : int, optional
        Level of detail to utilize in logging.
    verbose : int, optional
       Level of detail to include in console output.
    reconstruct: bool, optional
        Whether to store the structure of individuals. More computationally expensive, but allows usage outside the algorithm.
    minimization : bool, optional
        If True, the objective is to minimize the fitness function. If False, maximize it (default is True).
    fitness_function : str, optional
        The fitness function used for evaluating individuals (default is from gp_solve_parameters).
    initializer : str, optional
        The strategy for initializing the population (e.g., "grow", "full", "rhh").
    prob_const : float, optional
        The probability of a constant being chosen rather than a terminal in trees creation (default: 0.2).
    tree_functions : list, optional
        List of allowed functions that can appear in the trees. Check documentation for the available functions.
    tree_constants : list, optional
        List of constants allowed to appear in the trees.
    max_depth: int, optional
        Max depth for the SLIM GSGP trees.
    copy_parent: bool, optional
        Whether to copy the original parent when mutation is impossible (due to depth rules or mutation constraints).
    n_jobs : int, optional
        Number of parallel jobs to run (default is 1).
    tournament_size : int, optional
        Tournament size to utilize during selection. Only applicable if using tournament selection. (Default is 2)
    test_elite : bool, optional
        Whether to test the elite individual on the test set after each generation.


    Returns
    -------
        Individual
            Returns the best individual at the last generation.
    """

    # ================================
    #         Input Validation
    # ================================

    # Setting the log_path
    if log_path is None:
        log_path = os.path.join(os.getcwd(), "log", "slim_gsgp.csv")

    op, sig, trees = check_slim_version(slim_version=slim_version)

    validate_inputs(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pop_size=pop_size, n_iter=n_iter,
                    elitism=elitism, n_elites=n_elites, init_depth=init_depth, log_path=log_path, prob_const=prob_const,
                    tree_functions=tree_functions, tree_constants=tree_constants, log=log_level, verbose=verbose,
                    minimization=minimization, n_jobs=n_jobs, test_elite=test_elite, fitness_function=fitness_function,
                    initializer=initializer, tournament_size=tournament_size)

    # Checking that both ms bounds are numerical
    assert isinstance(ms_lower, (int, float)) and isinstance(ms_upper, (int, float)), \
        "Both ms_lower and ms_upper must be either int or float"

    if test_elite and (X_test is None or y_test is None):
        warnings.warn("If test_elite is True, a test dataset must be provided. test_elite has been set to False")
        test_elite = False

    if dataset_name is None:
        warnings.warn("No dataset name set. Using default value of dataset_1.")
        dataset_name = "dataset_1"

    # If so, create the ms callable
    ms = generate_random_uniform(ms_lower, ms_upper)

    if not isinstance(max_depth, int) and max_depth is not None:
        raise TypeError("max_depth value must be a int or None")

    if max_depth is not None:
        assert init_depth + 6 <= max_depth, f"max_depth must be at least {init_depth + 6}"


    # creating a list with the valid available fitness functions
    valid_fitnesses = list(fitness_function_options)

    # assuring the chosen fitness_function is valid
    assert fitness_function.lower() in fitness_function_options.keys(), \
        "fitness function must be: " + f"{', '.join(valid_fitnesses[:-1])} or {valid_fitnesses[-1]}" \
            if len(valid_fitnesses) > 1 else valid_fitnesses[0]

    # creating a list with the valid available initializers
    valid_initializers = list(initializer_options)

    # assuring the chosen initializer is valid
    assert initializer.lower() in initializer_options.keys(), \
        "initializer must be " + f"{', '.join(valid_initializers[:-1])} or {valid_initializers[-1]}" \
            if len(valid_initializers) > 1 else valid_initializers[0]

    # ================================
    #       Parameter Definition
    # ================================

    # setting the number of elites to 0 if no elitism is used
    if not elitism:
        n_elites = 0


    #   *************** SLIM_GSGP_PI_INIT ***************
    TERMINALS = get_terminals(X_train)

    slim_gsgp_pi_init["TERMINALS"] = TERMINALS
    try:
        slim_gsgp_pi_init["FUNCTIONS"] = {key: FUNCTIONS[key] for key in tree_functions}
    except KeyError as e:
        valid_functions = list(FUNCTIONS)
        raise KeyError(
            "The available tree functions are: " + f"{', '.join(valid_functions[:-1])} or "f"{valid_functions[-1]}"
            if len(valid_functions) > 1 else valid_functions[0])

    try:
        slim_gsgp_pi_init['CONSTANTS'] = {f"constant_{str(n).replace('-', '_')}": lambda _, num=n: torch.tensor(num)
                                          for n in tree_constants}
    except KeyError as e:
        valid_constants = list(CONSTANTS)
        raise KeyError(
            "The available tree constants are: " + f"{', '.join(valid_constants[:-1])} or "f"{valid_constants[-1]}"
            if len(valid_constants) > 1 else valid_constants[0])

    slim_gsgp_pi_init["init_pop_size"] = pop_size
    slim_gsgp_pi_init["init_depth"] = init_depth
    slim_gsgp_pi_init["p_c"] = prob_const

    #   *************** SLIM_GSGP_PARAMETERS ***************

    slim_gsgp_parameters["two_trees"] = trees
    slim_gsgp_parameters["operator"] = op

    slim_gsgp_parameters["p_m"] = 1 - slim_gsgp_parameters["p_xo"]
    slim_gsgp_parameters["pop_size"] = pop_size
    slim_gsgp_parameters["inflate_mutator"] = inflate_mutation(
        FUNCTIONS= slim_gsgp_pi_init["FUNCTIONS"],
        TERMINALS= slim_gsgp_pi_init["TERMINALS"],
        CONSTANTS= slim_gsgp_pi_init["CONSTANTS"],
        two_trees=slim_gsgp_parameters['two_trees'],
        operator=slim_gsgp_parameters['operator'],
        sig=sig
    )
    slim_gsgp_parameters["initializer"] = initializer_options[initializer]
    slim_gsgp_parameters["ms"] = ms
    slim_gsgp_parameters['p_inflate'] = p_inflate
    slim_gsgp_parameters['p_deflate'] = 1 - slim_gsgp_parameters['p_inflate']
    slim_gsgp_parameters["copy_parent"] = copy_parent
    slim_gsgp_parameters["seed"] = seed

    if minimization:
        slim_gsgp_parameters["selector"] = tournament_selection_min(tournament_size)
        slim_gsgp_parameters["find_elit_func"] = get_best_min
    else:
        slim_gsgp_parameters["selector"] = tournament_selection_max(tournament_size)
        slim_gsgp_parameters["find_elit_func"] = get_best_max


    #   *************** SLIM_GSGP_SOLVE_PARAMETERS ***************

    slim_gsgp_solve_parameters["log"] = log_level
    slim_gsgp_solve_parameters["verbose"] = verbose
    slim_gsgp_solve_parameters["log_path"] = log_path
    slim_gsgp_solve_parameters["elitism"] = elitism
    slim_gsgp_solve_parameters["n_elites"] = n_elites
    slim_gsgp_solve_parameters["n_iter"] = n_iter
    slim_gsgp_solve_parameters['run_info'] = [slim_version, UNIQUE_RUN_ID, dataset_name]
    slim_gsgp_solve_parameters["ffunction"] = fitness_function_options[fitness_function]
    slim_gsgp_solve_parameters["reconstruct"] = reconstruct
    slim_gsgp_solve_parameters["max_depth"] = max_depth
    slim_gsgp_solve_parameters["n_jobs"] = n_jobs
    slim_gsgp_solve_parameters["test_elite"] = test_elite

    # ================================
    #       Running the Algorithm
    # ================================

    optimizer = SLIM_GSGP(
        pi_init=slim_gsgp_pi_init,
        **slim_gsgp_parameters
    )

    optimizer.solve(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        curr_dataset=dataset_name,
        **slim_gsgp_solve_parameters
    )

    log_settings(
        path=os.path.join(os.getcwd(), "log", "slim_settings.csv"),
        settings_dict=[slim_gsgp_solve_parameters,
                       slim_gsgp_parameters,
                       slim_gsgp_pi_init,
                       settings_dict],
        unique_run_id=UNIQUE_RUN_ID
    )

    optimizer.elite.version = slim_version

    return optimizer.elite


if __name__ == "__main__":
    from slim_gsgp.datasets.data_loader import load_resid_build_sale_price
    from slim_gsgp.utils.utils import train_test_split, show_individual


    for ds in ["resid_build_sale_price"]:

        for s in range(30):

            X, y = load_resid_build_sale_price(X_y=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=s)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=s)

            #X_train, X_val, y_train, y_val = train_test_split(X, y, p_test=0.3, seed=s)

            for algorithm in ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]:

                final_tree = slim(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
                                  dataset_name=ds, slim_version=algorithm, max_depth=None, pop_size=100, n_iter=10, seed=s, p_inflate=0.2,
                                log_path=os.path.join(os.getcwd(),
                                                                "log", f"test_{ds}-size.csv"),
                                   reconstruct=True, n_jobs=1)

                #print(show_individual(final_tree, operator='sum'))
                #predictions = final_tree.predict(data=X_test, slim_version=algorithm)
                #print(float(rmse(y_true=y_test, y_pred=predictions)))
