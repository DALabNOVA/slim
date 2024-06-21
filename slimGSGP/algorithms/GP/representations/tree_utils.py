"""
Utility functions and tree operations for genetic programming.
"""

import random

import numpy as np
import torch


def bound_value(vector, min_val, max_val):
    """
    Constrains the values within a specific range.

    Parameters
    ----------
    vector : torch.Tensor
        Input tensor to be bounded.
    min_val : float
        Minimum value for bounding.
    max_val : float
        Maximum value for bounding.

    Returns
    -------
    torch.Tensor
        Tensor with values bounded between min_val and max_val.
    """
    return torch.clamp(vector, min_val, max_val)


def flatten(data):
    """
    Flattens a nested tuple structure.

    Parameters
    ----------
    data : tuple
        Input nested tuple data structure.

    Yields
    ------
    object
        Flattened data element by element.
    """
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data


def create_grow_random_tree(
    depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0.3, first_call=True
):
    """
    Generates a random tree using the Grow method with a specified depth.

    Parameters
    ----------
    depth : int
        Maximum depth of the tree to be created.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    p_c : float, optional
        Probability of choosing a constant node. Default is 0.3.
    first_call : bool, optional
        Variable that controls whether or not the function is being called for the first time. Default is True.

    Returns
    -------
    tuple
        The generated tree according to the specified parameters.
    """
    if p_c > 0:
        p_terminal = (len(TERMINALS) + len(CONSTANTS)) / (
            len(TERMINALS) + len(CONSTANTS) + len(FUNCTIONS)
        )
    else:
        p_terminal = len(TERMINALS) / (len(TERMINALS) + len(FUNCTIONS))

    if (depth <= 1 or random.random() < p_terminal) and not first_call:
        if random.random() > p_c:
            node = np.random.choice(list(TERMINALS.keys()))
        else:
            node = np.random.choice(list(CONSTANTS.keys()))
    else:
        node = np.random.choice(list(FUNCTIONS.keys()))
        if FUNCTIONS[node]["arity"] == 2:
            left_subtree = create_grow_random_tree(
                depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, False
            )
            right_subtree = create_grow_random_tree(
                depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, False
            )
            node = (node, left_subtree, right_subtree)
        else:
            left_subtree = create_grow_random_tree(
                depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, False
            )
            node = (node, left_subtree)
    return node


def create_full_random_tree(depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0.3):
    """
    Generates a full random tree with a specified depth.

    Parameters
    ----------
    depth : int
        Maximum depth of the tree to be created.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    p_c : float, optional
        Probability of choosing a function node. Default is 0.3.

    Returns
    -------
    tuple
        The generated full tree based on the specified parameters.
    """
    if depth <= 1:
        if random.random() > p_c:
            node = np.random.choice(list(TERMINALS.keys()))
        else:
            node = np.random.choice(list(CONSTANTS.keys()))
    else:
        node = np.random.choice(list(FUNCTIONS.keys()))
        if FUNCTIONS[node]["arity"] == 2:
            left_subtree = create_full_random_tree(
                depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c
            )
            right_subtree = create_full_random_tree(
                depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c
            )
            node = (node, left_subtree, right_subtree)
        else:
            left_subtree = create_full_random_tree(
                depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c
            )
            node = (node, left_subtree)
    return node


def random_subtree(FUNCTIONS):
    """
    Selects a random subtree from a given tree.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.

    Returns
    -------
    function
        A function that selects a random subtree from the input tree.
    """

    def random_subtree_picker(tree, first_call=True, num_of_nodes=None):
        if isinstance(tree, tuple):
            current_number_of_nodes = (
                num_of_nodes if first_call else len(list(flatten(tree)))
            )
            if FUNCTIONS[tree[0]]["arity"] == 2:
                if first_call:
                    subtree_exploration = (
                        1
                        if random.random()
                        < len(list(flatten(tree[1]))) / (current_number_of_nodes - 1)
                        else 2
                    )
                else:
                    p = random.random()
                    subtree_exploration = (
                        0
                        if p < 1 / current_number_of_nodes
                        else (
                            1
                            if p < len(list(flatten(tree[1]))) / current_number_of_nodes
                            else 2
                        )
                    )
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                subtree_exploration = (
                    1
                    if first_call
                    else (0 if random.random() < 1 / current_number_of_nodes else 1)
                )

            if subtree_exploration == 0:
                return tree
            elif subtree_exploration == 1:
                return (
                    random_subtree_picker(tree[1], False)
                    if isinstance(tree[1], tuple)
                    else tree[1]
                )
            elif subtree_exploration == 2:
                return (
                    random_subtree_picker(tree[2], False)
                    if isinstance(tree[2], tuple)
                    else tree[2]
                )
        else:
            return tree

    return random_subtree_picker


def substitute_subtree(FUNCTIONS):
    """
    Substitutes a specific subtree in a tree with a new subtree.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.

    Returns
    -------
    function
        A function that substitutes a subtree in the input tree.
    """

    def substitute(tree, target_subtree, new_subtree):
        if tree == target_subtree:
            return new_subtree
        elif isinstance(tree, tuple):
            if FUNCTIONS[tree[0]]["arity"] == 2:
                return (
                    tree[0],
                    substitute(tree[1], target_subtree, new_subtree),
                    substitute(tree[2], target_subtree, new_subtree),
                )
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                return tree[0], substitute(tree[1], target_subtree, new_subtree)
        else:
            return tree

    return substitute


def tree_pruning(TERMINALS, CONSTANTS, FUNCTIONS, p_c=0.3):
    """
    Reduces both sides of a tree to a specific depth.

    Parameters
    ----------
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    p_c : float, optional
        Probability of choosing a constant node. Default is 0.3.

    Returns
    -------
    function
        A function that prunes the tree to the specified depth.
    """

    def pruning(tree, target_depth):
        if target_depth <= 1 and tree not in TERMINALS:
            return (
                np.random.choice(list(TERMINALS.keys()))
                if random.random() > p_c
                else np.random.choice(list(CONSTANTS.keys()))
            )
        elif not isinstance(tree, tuple):
            return tree
        if FUNCTIONS[tree[0]]["arity"] == 2:
            new_left_subtree = pruning(tree[1], target_depth - 1)
            new_right_subtree = pruning(tree[2], target_depth - 1)
            return tree[0], new_left_subtree, new_right_subtree
        elif FUNCTIONS[tree[0]]["arity"] == 1:
            new_left_subtree = pruning(tree[1], target_depth - 1)
            return tree[0], new_left_subtree

    return pruning


def tree_depth(FUNCTIONS):
    """
    Calculates the depth of a given tree.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.

    Returns
    -------
    function
        A function that calculates the depth of the input tree.
    """

    def depth(tree):
        if not isinstance(tree, tuple):
            return 1
        else:
            if FUNCTIONS[tree[0]]["arity"] == 2:
                left_depth = depth(tree[1])
                right_depth = depth(tree[2])
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                left_depth = depth(tree[1])
                right_depth = 0
            return 1 + max(left_depth, right_depth)

    return depth
