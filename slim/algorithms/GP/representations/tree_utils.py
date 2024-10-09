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
        A Tensor with values bounded between min_val and max_val.
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
        Flattened data element by element. If data is not a tuple, returns the original data itself.
    """
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data


def create_grow_random_tree(depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0.3, first_call=True):
    """
    Generates a random tree representation using the Grow method with a maximum specified depth.

    Utilizes recursion to call itself on progressively smaller depths to form the whole tree, until the leaf nodes.

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
        Variable that controls whether the function is being called for the first time. Default is True.

    Returns
    -------
    tuple
        The generated tree representation according to the specified parameters.
    str
        The terminal or constant node selected, depending on depth and random probabilities.
    """
    # defining the probability for a terminal node to be selected, if the probability of constants is not 0
    if p_c > 0:
        p_terminal = (len(TERMINALS) + len(CONSTANTS)) / (len(TERMINALS) + len(CONSTANTS) + len(FUNCTIONS))
    else:
        p_terminal = len(TERMINALS) / (len(TERMINALS) + len(FUNCTIONS))

    # if a terminal is selected (or depth is 1) and its not the first call of the create_grow_random_tree function
    if (depth <= 1 or random.random() < p_terminal) and not first_call:
        # choosing between a constant or a terminal
        if random.random() > p_c:
            node = np.random.choice(list(TERMINALS.keys()))
        else:
            node = np.random.choice(list(CONSTANTS.keys()))

    # if a function is selected
    else:
        # selecting a random function
        node = np.random.choice(list(FUNCTIONS.keys()))
        # creating the tree based on the selected function's arity
        if FUNCTIONS[node]["arity"] == 2:
            left_subtree = create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, False)
            right_subtree = create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, False)
            node = (node, left_subtree, right_subtree)
        else:
            left_subtree = create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c, False)
            node = (node, left_subtree)
    return node


def create_full_random_tree(depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=0.3):
    """
    Generates a full random tree representation with a specified depth.

    Utilizes recursion to call itself on progressively smaller depths to form the whole tree, until the leaf nodes.

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

    Returns
    -------
    tuple
        The generated tree representation according to the specified parameters.
    str
        The terminal or constant node selected, depending on depth and random probabilities.
    """
    # if the maximum depth is 1, choose a terminal node
    if depth <= 1:
        # choosing between a terminal or a constant to be the terminal node
        if random.random() > p_c:
            node = np.random.choice(list(TERMINALS.keys()))
        else:
            node = np.random.choice(list(CONSTANTS.keys()))
    # if the depth isn't one, choose a random function
    else:
        node = np.random.choice(list(FUNCTIONS.keys()))
        # building the tree based on the arity of the chosen function
        if FUNCTIONS[node]["arity"] == 2:
            left_subtree = create_full_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c)
            right_subtree = create_full_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c)
            node = (node, left_subtree, right_subtree)
        else:
            left_subtree = create_full_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS, p_c)
            node = (node, left_subtree)
    return node


def random_subtree(FUNCTIONS):
    """
    Creates a function that selects a random subtree from a given tree representation.

    This function generates another function that traverses a tree representation to randomly
    select a subtree based on the arity of the functions within the tree.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.

    Returns
    -------
    Callable
        A function that selects a random subtree from the given tree representation.

        This function navigates the tree representation recursively, choosing a subtree based on
        probabilities determined by the overall representation of the tree.

        Parameters
        ----------
        tree : tuple
            The tree representation from which to select a subtree.
        first_call : bool, optional
            Indicates whether this is the initial call to the function. Defaults to True.
        num_of_nodes : int, optional
            The total number of nodes in the tree. Used to calculate probabilities.

        Returns
        -------
        tuple
            The randomly selected subtree (or the original node if not applicable).

    Notes
    -----
    The returned function traverses the tree representation recursively, selecting subtrees based on random
    probabilities influenced by the representation of the tree.
    """
    def random_subtree_picker(tree, first_call=True, num_of_nodes=None):
        """
        Selects a random subtree from the given tree representation.

        This function navigates the tree representation recursively, choosing a subtree based on
        probabilities determined by the overall representation of the tree.

        Parameters
        ----------
        tree : tuple
            The tree representation from which to select a subtree.
        first_call : bool, optional
            Indicates whether this is the initial call to the function. Defaults to True.
        num_of_nodes : int, optional
            The total number of nodes in the tree. Used to calculate probabilities.

        Returns
        -------
        tuple
            The randomly selected subtree (or the original node if not applicable).
        """
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
    Generates a function that substitutes a specific subtree in a tree representation with a new subtree.

    This function returns another function that can recursively traverse a tree representation to replace
    occurrences of a specified subtree with a new one, maintaining the representation and
    validity of the original tree.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.

    Returns
    -------
    Callable
        A function that substitutes a specified subtree within the given tree representation with a new subtree.

        This function recursively searches for occurrences of the target subtree within the tree
        representation and replaces it with the new subtree when found. If the original tree
        representation is a terminal or equal to the new one, return it.

        Parameters
        ----------
        tree : tuple or str
            The tree representation in which to perform the substitution. Can be a terminal.
        target_subtree : tuple or str
            The subtree to be replaced.
        new_subtree : tuple or str
            The subtree to insert in place of the target subtree.

        Returns
        -------
        tuple
            The modified tree representation with the target subtree replaced by the new subtree.
        str
            The new tree leaf node if the original is a leaf.

    Notes
    -----
    The returned function performs replacements while preserving the tree structure based on
    the arity of the function nodes.
    """

    def substitute(tree, target_subtree, new_subtree):
        """
        Substitutes a specified subtree within the given tree representation with a new subtree.

        This function recursively searches for occurrences of the target subtree within the tree
        representation and replaces it with the new subtree when found. If the original tree
        representation is a terminal or equal to the new one, return it.

        Parameters
        ----------
        tree : tuple or str
            The tree representation in which to perform the substitution. Can be a terminal.
        target_subtree : tuple or str
            The subtree to be replaced.
        new_subtree : tuple or str
            The subtree to insert in place of the target subtree.

        Returns
        -------
        tuple
            The modified tree representation with the target subtree replaced by the new subtree.
        str
            The new tree leaf node if the original is a leaf.
        """
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
    Generates a function that reduces both sides of a tree representation to a specific depth.

    This function returns another function that can prune a given tree representation to a
    specified depth by replacing nodes with terminals or constants based on a defined probability.

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
    Callable
        A function ('pruning') that prunes the given tree representation to the specified depth.

        This function replaces nodes in the tree representation with terminals or constants
        if the target depth is reached, ensuring the tree representation does not exceed the
        specified depth.

        Parameters
        ----------
        tree : tuple or str
            The tree representation to be pruned.
        target_depth : int
            The depth to which the tree representation should be pruned.

        Returns
        -------
        tuple
            The pruned tree representation, which may consist of terminals, constants, or
            a modified subtree.
        str
            The pruned tree if it is a leaf.
    """
    def pruning(tree, target_depth):
        """
        Prunes the given tree representation to the specified depth.

        This function replaces nodes in the tree representation with terminals or constants
        if the target depth is reached, ensuring the tree representation does not exceed the
        specified depth.

        Parameters
        ----------
        tree : tuple or str
            The tree representation to be pruned.
        target_depth : int
            The depth to which the tree representation should be pruned.

        Returns
        -------
        tuple
            The pruned tree representation, which may consist of terminals, constants, or
            a modified subtree.
        str
            The pruned tree if it is a leaf.
        """
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
    Generates a function that calculates the depth of a given tree representation.

    This function returns another function that can be used to compute the depth
    of a tree representation, which is defined as the length of the longest path
    from the root node to a leaf node.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree representation.

    Returns
    -------
    Callable
        A function ('depth') that calculates the depth of the given tree.

        This function determines the depth by recursively computing the maximum
        depth of the left and right subtrees and adding one for the current node.

        Parameters
        ----------
        tree : tuple or str
            The tree representation for which to calculate the depth. It can also be
            a terminal node represented as a string.

        Returns
        -------
        int
            The depth of the tree.

    Notes
    -----
    The returned function traverses the tree representation recursively, determining
    the depth based on the max of the subtree depths.
    """
    def depth(tree):
        """
        Calculates the depth of the given tree.

        This function determines the depth by recursively computing the maximum
        depth of the left and right subtrees and adding one for the current node.

        Parameters
        ----------
        tree : tuple or str
            The tree representation for which to calculate the depth. It can also be
            a terminal node represented as a string.

        Returns
        -------
        int
            The depth of the tree.
        """
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


def _execute_tree(repr_, X, FUNCTIONS, TERMINALS, CONSTANTS):
    """
    Evaluates a tree genotype on input vectors.

    Parameters
    ----------
    repr_ : tuple
        Tree representation.

    FUNCTIONS : dict
        Dictionary of allowed functions in the tree representation.

    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree representation.

    CONSTANTS : dict
        Dictionary of constant values allowed in the tree representation.

    Returns
    -------
    float
        Output of the evaluated tree representation.
    """
    if isinstance(repr_, tuple):  # If it's a function node
        function_name = repr_[0]
        if FUNCTIONS[function_name]["arity"] == 2:
            left_subtree, right_subtree = repr_[1], repr_[2]
            left_result = _execute_tree(left_subtree, X, FUNCTIONS, TERMINALS,
                                        CONSTANTS)  # equivalent to Tree(left_subtree).apply_tree(inputs) if no parallelization were used
            right_result = _execute_tree(right_subtree, X, FUNCTIONS, TERMINALS,
                                         CONSTANTS)  # equivalent to Tree(right_subtree).apply_tree(inputs) if no parallelization were used
            output = FUNCTIONS[function_name]["function"](
                left_result, right_result
            )
        else:
            left_subtree = repr_[1]
            left_result = _execute_tree(left_subtree, X, FUNCTIONS, TERMINALS,
                                        CONSTANTS)  # equivalent to Tree(left_subtree).apply_tree(inputs) if no parallelization were used
            output = FUNCTIONS[function_name]["function"](left_result)

        return bound_value(output, -1e12, 1e12)

    else:  # If it's a terminal node
        if repr_ in TERMINALS:
            return X[:, TERMINALS[repr_]]
        elif repr_ in CONSTANTS:
            return CONSTANTS[repr_](None)
