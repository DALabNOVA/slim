"""
This script runs the StandardGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import uuid
import os
from slim.algorithms.GP.gp import GP
from slim.algorithms.GP.operators.mutators import mutate_tree_subtree
from slim.algorithms.GP.representations.tree_utils import tree_depth, tree_pruning
from slim.config.gp_config import *
from slim.selection.selection_algorithms import tournament_selection_max, tournament_selection_min
from slim.utils.logger import log_settings
from slim.utils.utils import (get_terminals, validate_inputs, validate_constants_dictionary,
                              validate_functions_dictionary, get_best_max, get_best_min)

# todo: would not be better to first log the settings and then perform the algorithm?
def gp(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
       dataset_name: str = None, pop_size: int = 100, n_iter: int = 1000, p_xo: float = 0.8,
       elitism: bool = True, n_elites: int = 1, max_depth: int = 17, init_depth: int = 6,
       log_path: str = os.path.join(os.getcwd(), "log", "gp.csv"), seed: int = 42,
       log: int = 1,
       verbose: int = 1,
       minimization: bool = True,
       fitness_function: str = "rmse",
       initializer: str = "rhh",
       n_jobs: int = 1,
       prob_const: float = 0.2,
       tree_functions: list = list(FUNCTIONS.keys()),
       tree_constants: list = list(CONSTANTS.keys())):
    """
    Main function to execute the StandardGP algorithm on specified datasets

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
    max_depth : int, optional
        The maximum depth for the GP trees.
    init_depth : int, optional
        The depth value for the initial GP trees population.
    log_path : str, optional
        The path where is created the log directory where results are saved.
    seed : int, optional
        Seed for the randomness

    Returns
    -------
    Tree
        Returns the best individual at the last generation.
    """

    validate_inputs(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pop_size=pop_size, n_iter=n_iter,
                    elitism=elitism, n_elites=n_elites, init_depth=init_depth, log_path=log_path, prob_const=prob_const,
                    tree_functions=tree_functions, tree_constants=tree_constants)

    assert 0 <= p_xo <= 1, "p_xo must be a number between 0 and 1"

    assert isinstance(max_depth, int), "Input must be a int"

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



    if not elitism:
        n_elites = 0

    unique_run_id = uuid.uuid1()

    algo = "StandardGP"
    gp_solve_parameters['run_info'] = [algo, unique_run_id, dataset_name]

    # GP PI INIT
    TERMINALS = get_terminals(X_train)
    gp_pi_init["TERMINALS"] = TERMINALS
    try:
        gp_pi_init["FUNCTIONS"] = {key: FUNCTIONS[key] for key in tree_functions}
    except KeyError as e:
        valid_functions = list(FUNCTIONS)
        raise KeyError(
            "The available tree functions are: " + f"{', '.join(valid_functions[:-1])} or "f"{valid_functions[-1]}"
            if len(valid_functions) > 1 else valid_functions[0])


    try:
        gp_pi_init["CONSTANTS"] = {key: CONSTANTS[key] for key in tree_constants}
    except KeyError as e:
        valid_constants = list(CONSTANTS)
        raise KeyError(
            "The available tree constants are: " + f"{', '.join(valid_constants[:-1])} or "f"{valid_constants[-1]}"
            if len(valid_constants) > 1 else valid_constants[0])


    gp_pi_init["p_c"] = prob_const

    gp_pi_init["init_pop_size"] = pop_size # TODO: why init pop_size != than rest?
    gp_pi_init["init_depth"] = init_depth

    # GP PARAMETERS
    gp_parameters["p_xo"] = p_xo
    gp_parameters["p_m"] = 1 - gp_parameters["p_xo"]
    gp_parameters["pop_size"] = pop_size
    gp_parameters["mutator"] = mutate_tree_subtree(
        gp_pi_init['init_depth'],  gp_pi_init["TERMINALS"], gp_pi_init['CONSTANTS'], gp_pi_init['FUNCTIONS'],
        p_c=gp_pi_init['p_c']
    )
    gp_parameters["initializer"] = initializer_options[initializer]

    # GP SOLVE PARAMETERS
    gp_solve_parameters["log"] = log
    gp_solve_parameters["verbose"] = verbose
    gp_solve_parameters["log_path"] = log_path
    gp_solve_parameters["elitism"] = elitism
    gp_solve_parameters["n_elites"] = n_elites
    gp_solve_parameters["max_depth"] = max_depth
    gp_solve_parameters["n_iter"] = n_iter
    gp_solve_parameters["tree_pruner"] = tree_pruning(
        TERMINALS=gp_pi_init['TERMINALS'], CONSTANTS=gp_pi_init['CONSTANTS'], FUNCTIONS=gp_pi_init['FUNCTIONS'],
        p_c=gp_pi_init["p_c"]
    )
    gp_solve_parameters['depth_calculator'] = tree_depth(FUNCTIONS=gp_pi_init['FUNCTIONS'])
    gp_solve_parameters["ffunction"] = fitness_function_options[fitness_function]
    gp_solve_parameters["n_jobs"] = n_jobs

    if X_test is not None and y_test is not None:
        gp_solve_parameters["test_elite"] = True
    else:
        gp_solve_parameters["test_elite"] = False

    if minimization:
        gp_parameters["selector"] = tournament_selection_min(2)
        gp_parameters["find_elit_func"] = get_best_min
    else:
        gp_parameters["selector"] = tournament_selection_max(2)
        gp_parameters["find_elit_func"] = get_best_max


    optimizer = GP(pi_init=gp_pi_init, **gp_parameters, seed=seed)
    optimizer.solve(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        curr_dataset=dataset_name,
        **gp_solve_parameters
    )

    log_settings(
        path=log_path[:-4] + "_settings.csv",
        settings_dict=[gp_solve_parameters,
                       gp_parameters,
                       gp_pi_init,
                       settings_dict],
        unique_run_id=unique_run_id,
    )

    return optimizer.elite


if __name__ == "__main__":
    from slim.datasets.data_loader import load_merged_data
    from slim.utils.utils import train_test_split

    X, y = load_merged_data("resid_build_sale_price", X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

    final_tree = gp(X_train=X_train, y_train=y_train,
                    X_test=X_val, y_test=y_val,
                    dataset_name='resid_build_sale_price', pop_size=100, n_iter=1000, prob_const=0, fitness_function="rmse", n_jobs=2)

    final_tree.print_tree_representation()
    predictions = final_tree.predict(X_test)
    print(float(rmse(y_true=y_test, y_pred=predictions)))
