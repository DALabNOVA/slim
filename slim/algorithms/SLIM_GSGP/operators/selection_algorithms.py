"""
Tournament Selection Functions for Genetic Programming using PyTorch.
"""

import random

import numpy as np


def tournament_selection_min_slim(pool_size):
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
        Function to execute tournament selection for minimum fitness.
    """

    def ts(pop):
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmin([ind.fitness for ind in pool])]

    return ts


def tournament_selection_max_slim(pool_size):
    """
    Performs tournament selection to select an individual with the lowest fitness from a population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        Function to execute tournament selection for minimum fitness.
    """


    def ts(pop):
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmax([ind.fitness for ind in pool])]

    return ts

