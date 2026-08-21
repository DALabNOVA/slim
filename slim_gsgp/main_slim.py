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
from pathlib import Path

import torch

from slim_gsgp.algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from slim_gsgp.config.slim_config import (
    CONSTANTS,
    FUNCTIONS,
    fitness_function_options,
    initializer_options,
    settings_dict,
    slim_gsgp_parameters,
    slim_gsgp_pi_init,
    slim_gsgp_solve_parameters,
)
from slim_gsgp.utils.logger import log_settings
from slim_gsgp.utils.utils import (get_terminals, check_slim_version, validate_inputs, generate_random_uniform,
                                   get_best_min, get_best_max)
from slim_gsgp.algorithms.SLIM_GSGP.operators.mutators import inflate_mutation
from slim_gsgp.selection.selection_algorithms import tournament_selection_max, tournament_selection_min


ELITES = {}


def _format_options(options):
    if len(options) == 1:
        return options[0]
    return f"{', '.join(options[:-1])} or {options[-1]}"


def _choice_error(prefix, options):
    return options[0] if len(options) == 1 else prefix + _format_options(options)


def slim(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
         dataset_name: str = "dataset_1",
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
         tournament_size: int = 2,
         test_elite: bool = slim_gsgp_solve_parameters["test_elite"],
         sigmoid_scaling_factor: float = 1.0,
         lam: float = 0.01,
         balanced: bool = False,
         use_adaptive_inflate: bool = False,
         adaptive_candidate_batch: int = 1,
         parsimony_tolerance: float = 0.0):

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
    tournament_size : int, optional
        Tournament size to utilize during selection. Only applicable if using tournament selection. (Default is 2)
    test_elite : bool, optional
        Whether to test the elite individual on the test set after each generation.
    sigmoid_scaling_factor : float, optional
        Scaling factor for the sigmoid used in the ``sigmoid_rmse`` fitness function.
        Only relevant when ``fitness_function="sigmoid_rmse"`` (default is 1.0).
    lam : float, optional
        Semantic L2 regularization strength for the ``margin`` (MS-SLIM) fitness
        function (default is 0.01). See
        ``slim_gsgp.classification`` and ``MS_SLIM_formulation.md``.
    balanced : bool, optional
        If True, use class-balanced weights for margin-based losses and adaptive
        inflate (default is False).
    use_adaptive_inflate : bool, optional
        If True and ``fitness_function="margin"``, replace the random inflate mutation
        step with the exact optimal step for that loss (default is False). See
        ``slim_gsgp.classification.adaptive_inflate``.
    adaptive_candidate_batch : int, optional
        Number of candidate blocks scored by exact adaptive line search before
        retaining the best one. Valid only with ``use_adaptive_inflate``.
    parsimony_tolerance : float, optional
        Relative fitness tolerance within a tournament. Candidates within the
        tolerance of the best fitness are selected by smallest node count.
        This is direct structural pressure, distinct from semantic L2.

    Returns
    -------
        Individual
            Returns the best individual at the last generation.
    """

    # ================================
    #         Input Validation
    # ================================

    # Every invocation receives isolated settings. The imported config dictionaries
    # are defaults, not mutable process-wide run state.
    pi_init = dict(slim_gsgp_pi_init)
    parameters = dict(slim_gsgp_parameters)
    solve_parameters = dict(slim_gsgp_solve_parameters)
    function_options = dict(fitness_function_options)
    run_id = uuid.uuid4()

    slim_version = slim_version.upper()
    fitness_function = fitness_function.lower()
    initializer = initializer.lower()

    # Setting the log_path
    if log_path is None:
        log_path = os.path.join(os.getcwd(), "log", "slim_gsgp.csv")

    op, sig, trees = check_slim_version(slim_version=slim_version)

    validate_inputs(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pop_size=pop_size, n_iter=n_iter,
                    elitism=elitism, n_elites=n_elites, init_depth=init_depth, log_path=log_path, prob_const=prob_const,
                    tree_functions=tree_functions, tree_constants=tree_constants, log=log_level, verbose=verbose,
                    minimization=minimization, n_jobs=1, test_elite=test_elite, fitness_function=fitness_function,
                    initializer=initializer, tournament_size=tournament_size)

    # Checking that both ms bounds are numerical
    if not isinstance(ms_lower, (int, float)) or not isinstance(ms_upper, (int, float)):
        raise TypeError("Both ms_lower and ms_upper must be either int or float")

    if test_elite and (X_test is None or y_test is None):
        raise ValueError("test_elite=True requires both X_test and y_test")

    if dataset_name is None:
        dataset_name = "dataset_1"

    # If so, create the ms callable
    ms = generate_random_uniform(ms_lower, ms_upper)

    if not isinstance(max_depth, int) and max_depth is not None:
        raise TypeError("max_depth value must be a int or None")

    if max_depth is not None:
        if init_depth + 6 > max_depth:
            raise ValueError(f"max_depth must be at least {init_depth + 6}")


    # if using sigmoid_rmse, rebuild it with the requested scaling factor
    if fitness_function.lower() == "sigmoid_rmse":
        from slim_gsgp.evaluators.fitness_functions import sigmoid_rmse as _sigmoid_rmse
        function_options["sigmoid_rmse"] = _sigmoid_rmse(sigmoid_scaling_factor)

    # if using a margin-based loss, rebuild it with the requested regularization strength
    if fitness_function.lower() in ("margin", "logistic", "code_regression"):
        from slim_gsgp.classification.losses import margin_loss as _margin_loss, \
            logistic_loss as _logistic_loss, code_regression_loss as _code_regression_loss
        function_options["margin"] = _margin_loss(lam, balanced=balanced)
        function_options["logistic"] = _logistic_loss()
        function_options["code_regression"] = _code_regression_loss()

    # creating a list with the valid available fitness functions
    valid_fitnesses = list(function_options)

    # assuring the chosen fitness_function is valid
    if fitness_function not in function_options:
        raise ValueError(_choice_error("fitness function must be: ", valid_fitnesses))

    # creating a list with the valid available initializers
    valid_initializers = list(initializer_options)

    # assuring the chosen initializer is valid
    if initializer not in initializer_options:
        raise ValueError(_choice_error("initializer must be ", valid_initializers))

    # ================================
    #       Parameter Definition
    # ================================

    # setting the number of elites to 0 if no elitism is used
    if not elitism:
        n_elites = 0


    #   *************** SLIM_GSGP_PI_INIT ***************
    TERMINALS = get_terminals(X_train)

    pi_init["TERMINALS"] = TERMINALS
    try:
        pi_init["FUNCTIONS"] = {key: FUNCTIONS[key] for key in tree_functions}
    except KeyError as e:
        valid_functions = list(FUNCTIONS)
        raise KeyError(
            "The available tree functions are: " + f"{', '.join(valid_functions[:-1])} or "f"{valid_functions[-1]}"
            if len(valid_functions) > 1 else valid_functions[0])

    try:
        pi_init['CONSTANTS'] = {f"constant_{str(n).replace('-', '_')}": lambda _, num=n: torch.tensor(num)
                                          for n in tree_constants}
    except KeyError as e:
        valid_constants = list(CONSTANTS)
        raise KeyError(
            "The available tree constants are: " + f"{', '.join(valid_constants[:-1])} or "f"{valid_constants[-1]}"
            if len(valid_constants) > 1 else valid_constants[0])

    pi_init["init_pop_size"] = pop_size
    pi_init["init_depth"] = init_depth
    pi_init["p_c"] = prob_const

    #   *************** SLIM_GSGP_PARAMETERS ***************

    parameters["two_trees"] = trees
    parameters["operator"] = op

    if parameters["p_xo"] != 0:
        raise ValueError("SLIM crossover is not implemented; p_xo must be 0")
    parameters["p_m"] = 1
    parameters["pop_size"] = pop_size
    parameters["inflate_mutator"] = inflate_mutation(
        FUNCTIONS=pi_init["FUNCTIONS"],
        TERMINALS=pi_init["TERMINALS"],
        CONSTANTS=pi_init["CONSTANTS"],
        two_trees=parameters['two_trees'],
        operator=parameters['operator'],
        sig=sig
    )
    if use_adaptive_inflate:
        if fitness_function != "margin":
            raise ValueError(
                "use_adaptive_inflate requires fitness_function='margin'"
            )
        from slim_gsgp.classification.adaptive_inflate import adaptive_inflate as _adaptive_inflate
        parameters["inflate_mutator"] = _adaptive_inflate(
            parameters["inflate_mutator"], y_train=y_train, lam=lam, operator=op,
            balanced=balanced, candidate_batch=adaptive_candidate_batch,
        )
    parameters["initializer"] = initializer_options[initializer]
    parameters["ms"] = ms
    parameters['p_inflate'] = p_inflate
    parameters['p_deflate'] = 1 - parameters['p_inflate']
    parameters["copy_parent"] = copy_parent
    parameters["seed"] = seed

    if minimization:
        parameters["selector"] = tournament_selection_min(
            tournament_size, parsimony_tolerance=parsimony_tolerance,
        )
        parameters["find_elit_func"] = get_best_min
    else:
        parameters["selector"] = tournament_selection_max(tournament_size)
        parameters["find_elit_func"] = get_best_max


    #   *************** SLIM_GSGP_SOLVE_PARAMETERS ***************

    solve_parameters["log"] = log_level
    solve_parameters["verbose"] = verbose
    solve_parameters["log_path"] = log_path
    solve_parameters["elitism"] = elitism
    solve_parameters["n_elites"] = n_elites
    solve_parameters["n_iter"] = n_iter
    solve_parameters['run_info'] = [slim_version, run_id, dataset_name]
    solve_parameters["ffunction"] = function_options[fitness_function]
    solve_parameters["reconstruct"] = reconstruct
    solve_parameters["max_depth"] = max_depth
    solve_parameters["test_elite"] = test_elite

    # ================================
    #       Running the Algorithm
    # ================================

    optimizer = SLIM_GSGP(
        pi_init=pi_init,
        **parameters
    )

    optimizer.solve(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        curr_dataset=dataset_name,
        **solve_parameters
    )

    if log_level != 0:
        log_file = Path(log_path)
        log_settings(
            path=str(log_file.with_name(f"{log_file.stem}_settings.csv")),
            settings_dict=[solve_parameters, parameters, pi_init, settings_dict],
            unique_run_id=run_id,
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
                                   reconstruct=True)

                #print(show_individual(final_tree, operator='sum'))
                #predictions = final_tree.predict(data=X_test, slim_version=algorithm)
                #print(float(rmse(y_true=y_test, y_pred=predictions)))
