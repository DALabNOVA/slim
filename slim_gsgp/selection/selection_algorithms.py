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
Selection operator implementation.
"""

import random

import numpy as np


def tournament_selection_min(pool_size):
    """
    Returns a function that performs tournament selection to select an individual with the lowest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """

    def ts(pop):
        """
        Selects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmin([ind.fitness for ind in pool])]

    return ts


def tournament_selection_max(pool_size):
    """
    Returns a function that performs tournament selection to select an individual with the highest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """
    def ts(pop):
        """
        Selects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the highest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmax([ind.fitness for ind in pool])]

    return ts

