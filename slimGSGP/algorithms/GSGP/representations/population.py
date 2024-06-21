"""
Population Class for Evolutionary Computation with Tree Structures using PyTorch.
"""


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

    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for each individual in the population.

        Args:
            inputs: Input data for calculating semantics.
            testing: Boolean indicating if the calculation is for testing semantics.

        Returns:
            None
        """
        # Calculate semantics for each individual in the population
        [
            individual.calculate_semantics(inputs, testing)
            for individual in self.population
        ]

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

    def evaluate(self, ffunction, y):
        """
        Evaluate the population using a fitness function.

        Args:
            ffunction: Fitness function to evaluate the individuals.
            y: Expected output (target) values as a torch tensor.

        Returns:
            None
        """
        # Evaluate all individuals in the population
        [individual.evaluate(ffunction, y) for individual in self.population]

        # Store the fitness values of each individual
        self.fit = [individual.fitness for individual in self.population]
