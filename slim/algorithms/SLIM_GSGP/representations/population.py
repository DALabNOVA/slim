# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Population Class for SLIM GSGP using PyTorch.
"""
from slim.utils.utils import _evaluate_slim_individual
from joblib import Parallel, delayed

class Population:
    def __init__(self, population):
        """
        Initialize the Population with a list of individuals.

        Parameters
        ----------
        population : list
            The list of individuals in the population.

        Returns
        -------
        None
        """
        self.population = population
        self.size = len(population)
        self.nodes_count = sum([ind.nodes_count for ind in population])
        self.fit = None
        self.train_semantics = None
        self.test_semantics = None

    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for each individual in the population.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data for calculating semantics.
        testing : bool, optional
            Boolean indicating if the calculation is for testing semantics.

        Returns
        -------
        None
        """
        # computing the semantics for all the individuals in the population
        [
            individual.calculate_semantics(inputs, testing)
            for individual in self.population
        ]

        # computing testing semantics, if applicable
        if testing:
            # setting the population semantics to be a list with all the semantics of all individuals
            self.test_semantics = [
                individual.test_semantics for individual in self.population
            ]

        else:
            # setting the population semantics to be a list with all the semantics of all individuals
            self.train_semantics = [
                individual.train_semantics for individual in self.population
            ]

    def __len__(self):
        """
        Return the size of the population.

        Returns
        -------
        int
            Size of the population.
        """
        return self.size

    def __getitem__(self, item):
        """
        Get an individual from the population by index.

        Parameters
        ----------
        item : int
            Index of the individual to retrieve.

        Returns
        -------
        Individual
            The individual at the specified index.
        """
        return self.population[item]

    def evaluate_no_parall(self, ffunction, y, operator="sum"):
        """
        Evaluate the population using a fitness function (without parallelization).
        This function is not currently in use, but has been retained for potential future use
        at the developer's discretion.

        Parameters
        ----------
        ffunction : Callable
            Fitness function to evaluate the individuals.
        y : torch.Tensor
            Expected output (target) values.
        operator : str, optional
            Operator to apply to the semantics. Default is "sum".

        Returns
        -------
        None
        """
        # evaluating all the individuals in the population
        [
            individual.evaluate(ffunction, y, operator=operator)
            for individual in self.population
        ]
        # defining the fitness of the population to be a list with the fitnesses of all individuals in the population
        self.fit = [individual.fitness for individual in self.population]

    def evaluate(self, ffunction, y, operator="sum", n_jobs=1):
        """
        Evaluate the population using a fitness function.

        Parameters
        ----------
        ffunction : Callable
            Fitness function to evaluate the individuals.
        y : torch.Tensor
            Expected output (target) values.
        operator : str, optional
            Operator to apply to the semantics ("sum" or "prod"). Default is "sum".
        n_jobs : int, optional
            The maximum number of concurrently running jobs for joblib parallelization. Default is 1.

        Returns
        -------
        None
        """
        # Evaluates individuals' fitnesses
        self.fit = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_slim_individual)(individual, ffunction=ffunction, y=y, operator=operator
            ) for individual in self.population)

        # Assigning individuals' fitness as an attribute
        [self.population[i].__setattr__('fitness', f) for i, f in enumerate(self.fit)]

