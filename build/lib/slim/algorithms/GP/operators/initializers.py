"""
Initializers operator implementation.
"""

from slim.algorithms.GP.representations.tree_utils import (create_full_random_tree,
                                                           create_grow_random_tree)


def grow(size, depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0.3):
    """
    Generates a list of individuals with random trees for a GM4OS population using the Grow method.

    Parameters
    ----------
    size : int
        The total number of individuals to be generated for the population.
    depth : int
        The maximum depth of the trees.
    FUNCTIONS : dict
        The functions allowed in the trees.
    TERMINALS : dict
        The terminal symbols allowed in the trees.
    CONSTANTS : dict
        The constant values allowed in the trees.
    p_c : float, optional
        The probability of choosing a constant node during tree creation. Default is 0.3.
    Returns
    -------
    list
        A list of Individual objects containing random trees and input sets based on the parameters provided.
    """

    return [
        create_grow_random_tree(depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c)
        for _ in range(size)
    ]


def full(size, depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0.3):
    """
    Generates a list of individuals with random trees for a GM4OS population using the Full method.

    Parameters
    ----------
    size : int
        The total number of individuals to be generated for the population.
    depth : int
        The maximum depth of the trees.
    FUNCTIONS : dict
        The functions allowed in the trees.
    TERMINALS : dict
        The symbols allowed in the trees.
    CONSTANTS : dict
        The constant values allowed in the trees.
    p_c : float, optional
        The probability of choosing a constant node during tree creation. Default is 0.3.
    Returns
    -------
    list
        A list of Individual objects containing random trees and input sets based on the parameters provided.
    """

    return [
        create_full_random_tree(depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c)
        for _ in range(2, size + 1)
    ]


def rhh(size, depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0.3):
    """
    Generates a list of individuals with random trees for a GM4OS population using the ramped-half-half method.

    Parameters
    ----------
    size : int
        The total number of individuals to be generated for the population.
    depth : int
        The maximum depth of the trees.
    FUNCTIONS : dict
        The list of functions allowed in the trees.
    TERMINALS : dict
        The list of terminal symbols allowed in the trees.
    CONSTANTS : dict
        The list of constant values allowed in the trees.
    p_c : float, optional
        The probability of choosing a constant node during tree creation. Default is 0.3.
    Returns
    -------
    list
        A list of Individual objects containing random trees and input sets based on the parameters provided.
    """

    population = []

    inds_per_bin = size / (depth - 1)
    for curr_depth in range(2, depth + 1):

        population.extend(
            [
                create_full_random_tree(
                    curr_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c
                )
                for _ in range(int(inds_per_bin // 2))
            ]
        )

        population.extend(
            [
                create_grow_random_tree(
                    curr_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c
                )
                for _ in range(int(inds_per_bin // 2))
            ]
        )

    while len(population) < size:
        population.append(
            create_grow_random_tree(depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c)
        )

    return population
