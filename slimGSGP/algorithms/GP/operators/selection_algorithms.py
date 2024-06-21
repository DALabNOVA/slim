"""
Selection operator implementation.
"""

import random

import numpy as np


def tournament_selection_max(pool_size):
    """
    Performs tournament selection to select an individual with the highest fitness from a population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    function
        Function to execute tournament selection.
    """

    def ts(pop):
        pool = random.sample(pop.population, k=pool_size)
        return pool[np.argmax([ind.fitness for ind in pool])]

    return ts


def tournament_selection_min(pool_size):
    """
    Performs tournament selection to select an individual with the lowest fitness from a population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    function
        Function to execute tournament selection for minimum fitness.
    """

    def ts(pop):
        pool = random.sample(pop.population, k=pool_size)
        return pool[np.argmin([ind.fitness for ind in pool])]

    return ts
