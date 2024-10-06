"""
Population Class for Evolutionary Computation with Tree Structures using PyTorch.
"""

from joblib import Parallel, delayed
from slim.algorithms.GSGP.representations.tree_utils import _execute_tree


class Population:
    def __init__(self, pop):
        """
        Initialize the population with individuals.

        Args:
            pop: List of individuals in the population.
        """
        self.population = pop
        self.size = len(pop)
        self.nodes_count = sum([ind.nodes for ind in pop])

    def calculate_semantics(self, inputs, testing=False, logistic=False):
        """
        Calculate the semantics for each individual in the population.

        Args:
            inputs: Input data for calculating semantics.
            testing: Boolean indicating if the calculation is for testing semantics.
            logistic: Boolean indicating whether to apply a logistic function to the semantics.

        Returns:
            None
        """
        # Calculate semantics for each individual in the population in a sequential fashion
        [_execute_tree(individual, inputs=inputs, testing=testing, logistic=logistic) for individual in self.population]

        # Store the semantics based on whether it's for testing or training
        if testing:
            self.test_semantics = [
                individual.test_semantics for individual in self.population
            ]
        else:
            self.train_semantics = [
                individual.train_semantics for individual in self.population
            ]

    def __len__(self):
        """
        Return the size of the population.

        Returns:
            int: Size of the population.
        """
        return self.size

    def __getitem__(self, item):
        """
        Get an individual from the population by index.

        Args:
            item: Index of the individual to retrieve.

        Returns:
            Individual: The individual at the specified index.
        """
        return self.population[item]

    def evaluate(self, ffunction, y, n_jobs=1):
        """
        Evaluate the population using a fitness function.

        Args:
            ffunction: Fitness function to evaluate the individuals.
            y: Expected output (target) values as a torch tensor.
            n_jobs: The maximum number of concurrently running jobs for joblib parallelization.
        Returns:
            None
        """
        # Evaluate all individuals in the population in a parallel fashion
        self.fit = Parallel(n_jobs=n_jobs)(
            delayed(individual._evaluate_parallel)(
                ffunction, y
            ) for individual in self.population
        )

        # Assign individuals' fitness
        [self.population[i].__setattr__('fitness', f) for i, f in enumerate(self.fit)]
