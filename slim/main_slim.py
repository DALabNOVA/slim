"""
This script runs the SLIM_GSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import time
import uuid

from slim.algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from slim.config.slim_config import *
from slim.utils.logger import log_settings
from slim.utils.utils import get_terminals, check_slim_version, validate_inputs, generate_random_uniform
from slim.algorithms.SLIM_GSGP.operators.mutators import inflate_mutation
from typing import Callable

ELITES = {}
UNIQUE_RUN_ID = uuid.uuid1()


# todo: would not be better to first log the settings and then perform the algorithm?
# todo: update how the name is saved and make it coherent with the paper
def slim(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
         dataset_name: str = None, slim_version: str = "SLIM+SIG2", pop_size: int = 100,
         n_iter: int = 100, elitism: bool = True, n_elites: int = 1, init_depth: int = 6,
         ms: Callable = generate_random_uniform(0, 1), p_inflate: float = 0.5,
         log_path: str = os.path.join(os.getcwd(), "log", "slim.csv"), seed: int = 1):
    """
    Main function to execute the SLIM GSGP algorithm on specified datasets

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
    slim_version : list
        The version of SLIM-GSGP that needs to be run
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
    ms : Callable, optional
        A function that will generate the mutation step
    p_inflate : float, optional
        Probability to apply the inflate mutation
    log_path : str, optional
        The path where is created the log directory where results are saved.
    seed : int, optional
        Seed for the randomness

    Returns
    -------
      Tree
        Returns the best individual at the last generation.
    """
    op, sig, trees = check_slim_version(slim_version=slim_version)

    validate_inputs(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    pop_size=pop_size, n_iter=n_iter, elitism=elitism, n_elites=n_elites, init_depth=init_depth,
                    log_path=log_path)

    slim_gsgp_parameters["two_trees"] = trees
    slim_gsgp_parameters["operator"] = op

    TERMINALS = get_terminals(X_train)

    slim_gsgp_parameters["ms"] = ms
    slim_gsgp_parameters['p_inflate'] = p_inflate
    slim_gsgp_parameters['p_deflate'] = 1 - slim_gsgp_parameters['p_inflate']

    slim_gsgp_pi_init["TERMINALS"] = TERMINALS
    slim_gsgp_pi_init["init_pop_size"] = pop_size
    slim_gsgp_pi_init["init_depth"] = init_depth

    slim_gsgp_parameters["p_m"] = 1 - slim_gsgp_parameters["p_xo"]
    slim_gsgp_parameters["pop_size"] = pop_size
    slim_gsgp_parameters["inflate_mutator"] = inflate_mutation(
        FUNCTIONS=FUNCTIONS,
        TERMINALS=TERMINALS,
        CONSTANTS=CONSTANTS,
        two_trees=slim_gsgp_parameters['two_trees'],
        operator=slim_gsgp_parameters['operator'],
        sig=sig
    )

    slim_gsgp_solve_parameters["log_path"] = log_path
    slim_gsgp_solve_parameters["elitism"] = elitism
    slim_gsgp_solve_parameters["n_elites"] = n_elites
    slim_gsgp_solve_parameters["n_iter"] = n_iter
    slim_gsgp_solve_parameters['run_info'] = [slim_version, UNIQUE_RUN_ID, dataset_name]
    if X_test is not None and y_test is not None:
        slim_gsgp_solve_parameters["test_elite"] = True
    else:
        slim_gsgp_solve_parameters["test_elite"] = False

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

    return optimizer.elite


if __name__ == "__main__":
    from datasets.data_loader import load_ppb
    from slim.utils.utils import train_test_split, show_individual

    X, y = load_ppb(X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

    algorithm = "SLIM+SIG2"

    final_tree = slim(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
                      dataset_name='ppb', slim_version=algorithm, pop_size=100, n_iter=2)

    print(show_individual(final_tree, operator='sum'))
    predictions = final_tree.predict(data=X_test, slim_version=algorithm)
    print(float(rmse(y_true=y_test, y_pred=predictions)))
