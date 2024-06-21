"""
Population class implementation for evaluating genetic programming individuals.
"""


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

    def evaluate(self, ffunction, X, y):
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

        Returns
        -------
        None
            Attributes a fitness tensor to the population.
        """
        for individual in self.population:
            individual.evaluate(ffunction, X, y)

        self.fit = [individual.fitness for individual in self.population]
