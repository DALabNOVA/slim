"""
Population class implementation for evaluating genetic programming individuals.
"""
from joblib import Parallel, delayed
from slim.algorithms.GP.representations.tree_utils import _execute_tree


class Population:
    def __init__(self, pop):
        """
        Initializes a population of Trees.

        This constructor sets up the population with a list of Tree objects,
        calculating the size of the population and the total node count.

        Parameters
        ----------
        pop : List
            The list of individual Tree objects that make up the population.

        Returns
        -------
        None
        """
        self.population = pop
        self.size = len(pop)
        self.nodes_count = sum(ind.node_count for ind in pop)
        self.fit = None

    def evaluate(self, ffunction, X, y, n_jobs=1):
        """
        Evaluates the population given a certain fitness function, input data (X), and target data (y).

        Attributes a fitness tensor to the population.

        Parameters
        ----------
        ffunction : function
            Fitness function to evaluate the individuals.
        X : torch.Tensor
            The input data (which can be training or testing).
        y : torch.Tensor
            The expected output (target) values.
        n_jobs : int
            The maximum number of concurrently running jobs for joblib parallelization.

        Returns
        -------
        None
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
