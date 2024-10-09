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
Initializers operator implementation.
"""

from slim.algorithms.GP.representations.tree_utils import (create_full_random_tree,
                                                           create_grow_random_tree)

def grow(init_pop_size, init_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0.3):
    """
    Generates a list of individuals with random trees for a GM4OS population using the Grow method.

    Parameters
    ----------
    init_pop_size : int
        The total number of individuals to be generated for the population.
    init_depth : int
        The maximum depth of the trees.
    FUNCTIONS : dict
        The dictionary of functions allowed in the trees.
    TERMINALS : dict
        The dictionary of terminal symbols allowed in the trees.
    CONSTANTS : dict
        The dictionary of constant values allowed in the trees.
    p_c : float, optional
        The probability of choosing a constant node during tree creation. Default is 0.3.
    Returns
    -------
    list
        A list of Individual objects containing random trees and input sets based on the parameters provided.
    """

    return [
        create_grow_random_tree(init_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c)
        for _ in range(init_pop_size)
    ]


def full(init_pop_size, init_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0.3):
    """
    Generates a list of individuals with random trees for a GM4OS population using the Full method.

    Parameters
    ----------
    init_pop_size : int
        The total number of individuals to be generated for the population.
    init_depth : int
        The maximum depth of the trees.
    FUNCTIONS : dict
        The dictionary of functions allowed in the trees.
    TERMINALS : dict
        The dictionary of terminal symbols allowed in the trees.
    CONSTANTS : dict
        The dictionary of constant values allowed in the trees.
    p_c : float, optional
        The probability of choosing a constant node during tree creation. Default is 0.3.
    Returns
    -------
    list
        A list of Individual objects containing random trees and input sets based on the parameters provided.
    """

    return [
        create_full_random_tree(init_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c)
        for _ in range(2, init_pop_size + 1)
    ]


def rhh(init_pop_size, init_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0.3):
    """
    Generates a list of individuals with random trees for a GM4OS population using the ramped-half-half method.

    Parameters
    ----------
    init_pop_size : int
        The total number of individuals to be generated for the population.
    init_depth : int
        The maximum depth of the trees.
    FUNCTIONS : dict
        The dictionary of functions allowed in the trees.
    TERMINALS : dict
        The dictionary of terminal symbols allowed in the trees.
    CONSTANTS : dict
        The dictionary of constant values allowed in the trees.
    p_c : float, optional
        The probability of choosing a constant node during tree creation. Default is 0.3.
    Returns
    -------
    list
        A list of Individual objects containing random trees and input sets based on the parameters provided.
    """

    population = []

    inds_per_bin = init_pop_size / (init_depth - 1)

    for curr_depth in range(2, init_depth + 1):

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

    while len(population) < init_pop_size:
        population.append(
            create_grow_random_tree(init_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c)
        )

    return population
