"""
This script runs the SLIM_GSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import uuid
import os
from slim.algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from slim.config.slim_config import *
from slim.utils.logger import log_settings
from slim.utils.utils import (get_terminals, check_slim_version, validate_inputs, generate_random_uniform,
                              validate_constants_dictionary, validate_functions_dictionary, get_best_min, get_best_max)
from slim.algorithms.SLIM_GSGP.operators.mutators import inflate_mutation
from slim.algorithms.SLIM_GSGP.operators.selection_algorithms import (tournament_selection_max_slim,
                                                                      tournament_selection_min_slim)


ELITES = {}
UNIQUE_RUN_ID = uuid.uuid1()


def slim(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
         dataset_name: str = None, slim_version: str = "SLIM+SIG2", pop_size: int = 100,
         n_iter: int = 100, elitism: bool = True, n_elites: int = 1, init_depth: int = 6,
         ms_lower: float = 0, ms_upper: float = 1, p_inflate: float = 0.5,
         log_path: str = os.path.join(os.getcwd(), "log", "slim.csv"), seed: int = 1,
         log_level: int = 1,
         verbose: int = 1,
         reconstruct: bool = False,
         fitness_function: str = "rmse",
         initializer: str = "rhh",
         minimization: bool = True,
         prob_const: float = 0.2,
         tree_functions: dict = list(FUNCTIONS.keys()),
         tree_constants: dict = list(CONSTANTS.keys()),
         copy_parent: bool = False,
         max_depth: int = None,
         n_jobs: int = 1,
         test_elite: bool = True):
    """
    Main function to execute the SLIM GSGP algorithm on specified datasets.

    Args:
        X_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training output data.
        X_test (torch.Tensor, optional): Testing input data. Defaults to None.
        y_test (torch.Tensor, optional): Testing output data. Defaults to None.
        dataset_name (str, optional): Name of the dataset for logging purposes. Defaults to None.
        slim_version (list): The version of SLIM-GSGP to be run.
        pop_size (int, optional): Population size for the genetic programming algorithm. Defaults to 100.
        n_iter (int, optional): Number of iterations for the genetic programming algorithm. Defaults to 100.
        elitism (bool, optional): Whether to use elitism. Defaults to False.
        n_elites (int, optional): Number of elites. Defaults to 0.
        init_depth (int, optional): Depth of the initial GP trees population. Defaults to None.
        ms_lower (float, optional): Lower bound for the mutation step. Defaults to 0
        ms_upper (float, optional): Lower bound for the mutation step. Defaults to 1
        p_inflate (float, optional): Probability to apply the inflate mutation. Defaults to None.
        log_path (str, optional): Path where the log directory and results are saved. Defaults to None.
        seed (int, optional): Seed for randomness. Defaults to None.
        log (int, optional): Logging level.

    Returns:
        Tree: The best individual at the last generation.
    """

    # ================================
    #         Input Validation
    # ================================

    op, sig, trees = check_slim_version(slim_version=slim_version)

    validate_inputs(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pop_size=pop_size, n_iter=n_iter,
                    elitism=elitism, n_elites=n_elites, init_depth=init_depth, log_path=log_path, prob_const=prob_const,
                    tree_functions=tree_functions, tree_constants=tree_constants, log=log_level, verbose=verbose,
                    minimization=minimization, n_jobs=n_jobs, test_elite=test_elite, fitness_function=fitness_function,
                    initializer=initializer)

    # Checking that both ms bounds are numerical
    assert isinstance(ms_lower, (int, float)) and isinstance(ms_upper, (int, float)), \
        "Both ms_lower and ms_upper must be either int or float"

    # If so, create the ms callable
    ms = generate_random_uniform(ms_lower, ms_upper)

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
        slim_gsgp_pi_init["CONSTANTS"] = {key: CONSTANTS[key] for key in tree_constants}
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

    if minimization:
        slim_gsgp_parameters["selector"] = tournament_selection_min_slim(2)
        slim_gsgp_parameters["find_elit_func"] = get_best_min
    else:
        slim_gsgp_parameters["selector"] = tournament_selection_max_slim(2)
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
        **slim_gsgp_parameters,
        seed=seed
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
    from slim.datasets.data_loader import load_merged_data
    from slim.utils.utils import train_test_split, show_individual


    for ds in ["resid_build_sale_price"]:

        for s in range(30):

            X, y = load_merged_data(ds, X_y=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=s)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=s)

            #X_train, X_val, y_train, y_val = train_test_split(X, y, p_test=0.3, seed=s)

            for algorithm in ["SLIM*SIG1"]:

                final_tree = slim(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
                                  dataset_name=ds, slim_version=algorithm, max_depth=15, pop_size=100, n_iter=100, seed=s, p_inflate=0.2,
                                log_path=os.path.join(os.getcwd(),
                                                                "log", f"TEST_slim_postgrid_{ds}-size.csv"),
                                   reconstruct=True, n_jobs=2)

                print(show_individual(final_tree, operator='sum'))
                predictions = final_tree.predict(data=X_test, slim_version=algorithm)
                print(float(rmse(y_true=y_test, y_pred=predictions)))
