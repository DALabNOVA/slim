"""
This script runs the StandardGSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import uuid

from slim.algorithms.GSGP.gsgp import GSGP
from slim.config.gsgp_config import *
from slim.utils.logger import log_settings
from slim.utils.utils import get_terminals, validate_inputs
from typing import Callable


# todo: would not be better to first log the settings and then perform the algorithm?
def gsgp(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
         dataset_name: str = None, pop_size: int = 100, n_iter: int = 100, p_xo: float = 0.0, elitism: bool = True,
         n_elites: int = 1, init_depth: int = 8, ms: Callable = generate_random_uniform(0, 1),
         log_path: str = os.path.join(os.getcwd(), "log", "gsgp.csv"),
         seed: int = 1):
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
    ms : Callable, optional
        A function that will generate the mutation step
    log_path : str, optional
        The path where is created the log directory where results are saved.
    seed : int, optional
        Seed for the randomness


    Returns
    -------
       Tree
        Returns the best individual at the last generation.
    """

    validate_inputs(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    pop_size=pop_size, n_iter=n_iter, elitism=elitism, n_elites=n_elites, init_depth=init_depth,
                    log_path=log_path)
    assert 0 <= p_xo <= 1, "p_xo must be a number between 0 and 1"

    if not elitism:
        n_elites = 0

    unique_run_id = uuid.uuid1()

    algo_name = "StandardGSGP"
    gsgp_solve_parameters["run_info"] = [algo_name, unique_run_id, dataset_name]

    TERMINALS = get_terminals(X_train)

    gsgp_pi_init["TERMINALS"] = TERMINALS
    gsgp_pi_init["init_pop_size"] = pop_size
    gsgp_pi_init["init_depth"] = init_depth

    gsgp_parameters["p_xo"] = p_xo
    gsgp_parameters["p_m"] = 1 - gsgp_parameters["p_xo"]
    gsgp_parameters["pop_size"] = pop_size
    gsgp_parameters["ms"] = ms

    gsgp_solve_parameters["n_iter"] = n_iter
    gsgp_solve_parameters["log_path"] = log_path
    gsgp_solve_parameters["elitism"] = elitism
    gsgp_solve_parameters["n_elites"] = n_elites
    if X_test is not None and y_test is not None:
        gsgp_solve_parameters["test_elite"] = True
    else:
        gsgp_solve_parameters["test_elite"] = False

    optimizer = GSGP(pi_init=gsgp_pi_init, **gsgp_parameters, seed=seed)

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
    from datasets.data_loader import load_ppb
    from slim.utils.utils import train_test_split

    X, y = load_ppb(X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

    final_tree = gsgp(X_train=X_train, y_train=y_train,
                      X_test=X_val, y_test=y_val,
                      dataset_name='ppb', pop_size=100, n_iter=10)

    predictions = final_tree.predict(X_test)
    print(float(rmse(y_true=y_test, y_pred=predictions)))
