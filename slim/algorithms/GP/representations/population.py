"""
Population class implementation for evaluating genetic programming individuals.
"""
from joblib import Parallel, delayed
from slim.algorithms.GP.representations.tree_utils import _execute_tree


class Population:
    def __init__(self, pop):
        """
        Initializes the Population object.

        Parameters
        ----------
        pop : list
            List of individual objects that make up the population.
        """
        self.population = pop
        self.size = len(pop)
        self.nodes_count = sum(ind.node_count for ind in pop)

    def evaluate(self, ffunction, X, y, n_jobs=1):
        """
        Evaluates the population given a certain fitness function, input data (X), and target data (y).

        Parameters
        ----------
        ffunction : function
            Fitness function to evaluate the individual.

        X : torch.Tensor
            The input data (which can be training or testing).

        y : torch.Tensor
            The expected output (target) values.

        n_jobs : int
            The maximum number of concurrently running jobs for joblib parallelization.

        Returns
        -------
        None
            Attributes a fitness tensor to the population.
        """
        # Evaluates individuals' semantics
        y_pred = Parallel(n_jobs=n_jobs)(
            delayed(_execute_tree)(
                individual.repr_, X,
                individual.FUNCTIONS, individual.TERMINALS, individual.CONSTANTS
            ) for individual in self.population
        )
        # Evaluate fitnesses
        self.fit = [ffunction(y, y_pred_ind) for y_pred_ind in y_pred]
        # Assign individuals' fitness
        [self.population[i].__setattr__('fitness', f) for i, f in enumerate(self.fit)]
