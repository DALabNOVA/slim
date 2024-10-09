"""
This script runs the StandardGSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""

import uuid
import os
import warnings

from slim.algorithms.GSGP.gsgp import GSGP
from slim.config.gsgp_config import *
from slim.utils.logger import log_settings
from slim.utils.utils import get_terminals, validate_inputs, generate_random_uniform
from typing import Callable

# todo: would not be better to first log the settings and then perform the algorithm?
def gsgp(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
         dataset_name: str = None,
         pop_size: int = gsgp_parameters["pop_size"],
         n_iter: int = gsgp_solve_parameters["n_iter"],
         p_xo: float = gsgp_parameters["p_xo"],
         elitism: bool = gsgp_solve_parameters["elitism"],
         n_elites: int = gsgp_solve_parameters["n_elites"],
         init_depth: int = gsgp_pi_init["init_depth"],
         ms_lower: float = 0,
         ms_upper: float = 1,
         log_path: str = None,
         seed: int = gsgp_parameters["seed"],
         log_level: int = gsgp_solve_parameters["log"],
         verbose: int = gsgp_solve_parameters["verbose"],
         reconstruct: bool = gsgp_solve_parameters["reconstruct"],
         fitness_function: str = gsgp_solve_parameters["ffunction"],
         initializer: str = gsgp_parameters["initializer"],
         minimization: bool = True,
         prob_const: float = gsgp_pi_init["p_c"],
         tree_functions: list = list(FUNCTIONS.keys()),
         tree_constants: list = [float(key.replace("constant_", "").replace("_", "-")) for key in CONSTANTS],
         n_jobs: int = gsgp_solve_parameters["n_jobs"],
         tournament_size: int = 2,
         test_elite: bool = gsgp_solve_parameters["test_elite"]):
    """
    Main function to execute the Standard GSGP algorithm on specified datasets

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
    p_xo : float, optional
        The probability of crossover in the genetic programming algorithm. Must be a number between 0 and 1 (default is 0.8).
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
    n_jobs : int, optional
        Number of parallel jobs to run (default is 1).
    prob_const : float, optional
        The probability of introducing constants into the trees during evolution.
    tree_functions : list, optional
        List of allowed functions that can appear in the trees. Check documentation for the available functions.
    tree_constants : list, optional
        List of constants allowed to appear in the trees.
    tournament_size : int, optional
        Tournament size to utilize during selection. Only applicable if using tournament selection. (Default is 2)
    test_elite : bool, optional
        Whether to test the elite individual on the test set after each generation.

    Returns
    -------
        Tree
            Returns the best individual at the last generation.
    """
    # ================================
    #         Input Validation
    # ================================

    # Setting the log_path
    if log_path is None:
        log_path = os.path.join(os.getcwd(), "log", "gsgp.csv")

    validate_inputs(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pop_size=pop_size, n_iter=n_iter,
                    elitism=elitism, n_elites=n_elites, init_depth=init_depth, log_path=log_path, prob_const=prob_const,
                    tree_functions=tree_functions, tree_constants=tree_constants, log=log_level, verbose=verbose,
                    minimization=minimization, n_jobs=n_jobs, test_elite=test_elite, fitness_function=fitness_function,
                    initializer=initializer, tournament_size=tournament_size)

    if test_elite and (X_test is None or y_test is None):
        warnings.warn("If test_elite is True, a test dataset must be provided. test_elite has been set to False")
        test_elite = False

    if dataset_name is None:
        warnings.warn("No dataset name set. Using default value of dataset_1.")
        dataset_name = "dataset_1"

    # Checking that both ms bounds are numerical
    assert isinstance(ms_lower, (int, float)) and isinstance(ms_upper, (int, float)), \
        "Both ms_lower and ms_upper must be either int or float"

    # If so, create the ms callable
    ms = generate_random_uniform(ms_lower, ms_upper)

    # assuring the p_xo is valid
    assert 0 <= p_xo <= 1, "p_xo must be a number between 0 and 1"

    # creating a list with the valid available fitness functions
    valid_fitnesses = list(fitness_function_options)

    # assuring the chosen fitness_function is valid
    assert fitness_function.lower() in fitness_function_options.keys(), \
                    "fitness function must be: "+f"{', '.join(valid_fitnesses[:-1])} or {valid_fitnesses[-1]}"\
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

    # getting a unique run id for the settings logging
    unique_run_id = uuid.uuid1()

    # setting the algorithm name to standard gsgp for logging TODO: do I add this as a changeable thing?
    algo_name = "StandardGSGP"

    #   *************** GSGP_PI_INIT ***************

    # getting the terminals based on the training data
    TERMINALS = get_terminals(X_train)

    gsgp_pi_init["TERMINALS"] = TERMINALS
    try:
        gsgp_pi_init["FUNCTIONS"] = {key: FUNCTIONS[key] for key in tree_functions}
    except KeyError as e:
        valid_functions = list(FUNCTIONS)
        raise KeyError(
            "The available tree functions are: " + f"{', '.join(valid_functions[:-1])} or "f"{valid_functions[-1]}"
            if len(valid_functions) > 1 else valid_functions[0])


    try:
        gsgp_pi_init['CONSTANTS'] = {f"constant_{str(n).replace('-', '_')}": lambda _, num=n: torch.tensor(num)
                                     for n in tree_constants}
    except KeyError as e:
        valid_constants = list(CONSTANTS)
        raise KeyError(
            "The available tree constants are: " + f"{', '.join(valid_constants[:-1])} or "f"{valid_constants[-1]}"
            if len(valid_constants) > 1 else valid_constants[0])


    # setting up the configuration dictionaries based on the user given input
    gsgp_pi_init["init_pop_size"] = pop_size
    gsgp_pi_init["init_depth"] = init_depth
    gsgp_pi_init["p_c"] = prob_const

    #   *************** GSGP_PARAMETERS ***************

    gsgp_parameters["p_xo"] = p_xo
    gsgp_parameters["p_m"] = 1 - gsgp_parameters["p_xo"]
    gsgp_parameters["pop_size"] = pop_size
    gsgp_parameters["ms"] = ms
    gsgp_parameters["seed"] = seed

    gsgp_parameters["initializer"] = initializer_options[initializer]

    if minimization:
        gsgp_parameters["selector"] = tournament_selection_min(tournament_size)
        gsgp_parameters["find_elit_func"] = get_best_min
    else:
        gsgp_parameters["selector"] = tournament_selection_max(tournament_size)
        gsgp_parameters["find_elit_func"] = get_best_max

    #   *************** GSGP_SOLVE_PARAMETERS ***************

    # setting up the information of the run, for logging purposes
    gsgp_solve_parameters["run_info"] = [algo_name, unique_run_id, dataset_name]
    gsgp_solve_parameters["n_iter"] = n_iter
    gsgp_solve_parameters["log_path"] = log_path
    gsgp_solve_parameters["elitism"] = elitism
    gsgp_solve_parameters["n_elites"] = n_elites
    gsgp_solve_parameters["n_jobs"] = n_jobs
    gsgp_solve_parameters["test_elite"] = test_elite
    gsgp_solve_parameters["log"] = log_level
    gsgp_solve_parameters["verbose"] = verbose
    gsgp_solve_parameters["reconstruct"] = reconstruct
    gsgp_solve_parameters["ffunction"] = fitness_function_options[fitness_function]

    # ================================
    #       Running the Algorithm
    # ================================

    optimizer = GSGP(pi_init=gsgp_pi_init, **gsgp_parameters)

    optimizer.solve(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        curr_dataset=dataset_name,
        **gsgp_solve_parameters,
    )

    log_settings(
        path=log_path[:-4] + "_settings.csv",
        settings_dict=[gsgp_solve_parameters,
                       gsgp_parameters,
                       gsgp_pi_init,
                       settings_dict],
        unique_run_id=unique_run_id,
    )
    return optimizer.elite


if __name__ == "__main__":
    from slim.datasets.data_loader import load_resid_build_sale_price
    from slim.utils.utils import train_test_split

    X, y = load_resid_build_sale_price(X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

    final_tree = gsgp(X_train=X_train, y_train=y_train,
                      X_test=X_val, y_test=y_val,
                      dataset_name='resid_build_sale_price', pop_size=100, n_iter=1000, log_path=os.path.join(os.getcwd(),
                                                                "log", f"TESTING_GSGP.csv"), fitness_function="rmse", n_jobs=2)

    predictions = final_tree.predict(X_test)
    print(float(rmse(y_true=y_test, y_pred=predictions)))
