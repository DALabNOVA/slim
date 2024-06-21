"""
Mutator operator implementation.
"""

import random

import numpy as np
from slim.algorithms.GP.representations.tree_utils import (create_grow_random_tree,
                                                           random_subtree,
                                                           substitute_subtree)


# Function to perform mutation on a tree.
def mutate_tree_node(max_depth, TERMINALS, CONSTANTS, FUNCTIONS, p_c):
    """
    Generates a function for mutating a node within a tree.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree to consider during mutation.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    p_c : float
        Probability of choosing a constant node for mutation.
    Returns
    -------
    function
        A function for mutating a node within a tree according to the specified parameters.
    """

    def m_tn(tree):

        if max_depth <= 1 or not isinstance(tree, tuple):
            if random.random() > p_c:
                return np.random.choice(list(TERMINALS.keys()))
            else:
                return np.random.choice(list(CONSTANTS.keys()))

        if FUNCTIONS[tree[0]]["arity"] == 2:
            node_to_mutate = np.random.randint(0, 3)
        elif FUNCTIONS[tree[0]]["arity"] == 1:
            node_to_mutate = np.random.randint(0, 2)  #

        inside_m = mutate_tree_node(max_depth - 1, TERMINALS, CONSTANTS, FUNCTIONS, p_c)

        if node_to_mutate == 0:
            new_function = np.random.choice(list(FUNCTIONS.keys()))
            it = 0
            while (
                FUNCTIONS[tree[0]]["arity"] != FUNCTIONS[new_function]["arity"]
                or tree[0] == new_function
            ):
                new_function = np.random.choice(list(FUNCTIONS.keys()))
                it += 1
                if it >= 10:
                    new_function = tree[0]
                    break

            left_subtree = inside_m(tree[1])
            if FUNCTIONS[tree[0]]["arity"] == 2:
                right_subtree = inside_m(tree[2])
                return (new_function, left_subtree, right_subtree)
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                return (new_function, left_subtree)
        elif node_to_mutate == 1:
            left_subtree = inside_m(tree[1])
            if FUNCTIONS[tree[0]]["arity"] == 2:
                return (tree[0], left_subtree, tree[2])
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                return (tree[0], left_subtree)
        else:
            right_subtree = inside_m(tree[2])
            return (tree[0], tree[1], right_subtree)

    return m_tn


def mutate_tree_subtree(max_depth, TERMINALS, CONSTANTS, FUNCTIONS, p_c):
    """
    Generates a function for performing subtree mutation between two trees.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.

    Returns
    -------
    function
        A function for mutating subtrees between two trees based on the specified functions.
    """

    subtree_substitution = substitute_subtree(FUNCTIONS=FUNCTIONS)
    random_subtree_picker = random_subtree(FUNCTIONS=FUNCTIONS)

    def inner_mut(tree1, num_of_nodes=None):

        if isinstance(tree1, tuple):
            crossover_point_tree1 = random_subtree_picker(
                tree1, num_of_nodes=num_of_nodes
            )
            crossover_point_tree2 = create_grow_random_tree(
                max_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=p_c
            )
            new_tree1 = subtree_substitution(
                tree1, crossover_point_tree1, crossover_point_tree2
            )

            return new_tree1
        else:
            return tree1

    return inner_mut
