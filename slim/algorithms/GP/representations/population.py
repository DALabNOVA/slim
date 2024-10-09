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
