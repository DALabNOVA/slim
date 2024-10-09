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
Utility functions for Tree Evaluation and Mutation in GSGP.
"""

from slim.algorithms.GP.representations.tree import Tree
from slim.algorithms.GP.representations.tree_utils import bound_value
import torch

def _execute_tree(individual, inputs, testing=False, logistic=False):
    """
    Calculate the semantics for the tree, storing it as an attribute.

    Parameters
    ----------
    individual : Tree
        The tree individual whose semantics are being calculated.
    inputs : array-like
        Input data for calculating semantics.
    testing : bool, optional
        Indicates if the calculation is for testing semantics. Defaults to `False`.
    logistic : bool, optional
        Indicates if a logistic function should be applied to tree outputs. Defaults to `False`.

    Returns
    -------
    None

    Notes
    -----
    The individual to be executed must have the `reconstruct` attribute set to True; otherwise,
    its structure will not be available for evaluation.
    """
    # if the calculation is for testing semantics
    if testing and individual.test_semantics is None:
        # if the individual is a base (gp) tree, use apply_tree to compute its semantics
        if isinstance(individual.structure, tuple):
            individual.test_semantics = (
                torch.sigmoid(apply_tree(individual, inputs))
                if logistic
                else apply_tree(individual, inputs)
            )
        # if not, use the operator (mutation or crossover) with the base trees to compute the semantics
        else:
            individual.test_semantics = individual.structure[0](
                *individual.structure[1:], testing=True
            )

    # if the calculation is for training semantics
    elif individual.train_semantics is None:
        # if the individual is a base (gp) tree, use apply_tree to compute its semantics
        if isinstance(individual.structure, tuple):
            individual.train_semantics = (
                torch.sigmoid(apply_tree(individual, inputs))
                if logistic
                else apply_tree(individual, inputs)
            )

        # if not, use the operator (mutation or crossover) with the base trees to compute the semantics
        else:
            individual.train_semantics = individual.structure[0](
                *individual.structure[1:], testing=False
            )

def apply_tree(tree, inputs):
    """
    Evaluates the tree on input vectors.

    Parameters
    ----------
    tree : Tree
        The tree structure to be evaluated.
    inputs : torch.Tensor
        Input vectors x and y.

    Returns
    -------
    torch.Tensor
        Output of the evaluated tree.
    """
    if isinstance(tree.structure, tuple):  # If it's a function node
        function_name = tree.structure[0]
        if tree.FUNCTIONS[function_name]["arity"] == 2:
            left_subtree, right_subtree = tree.structure[1], tree.structure[2]
            left_subtree = Tree(left_subtree)
            right_subtree = Tree(right_subtree)
            left_result = left_subtree.apply_tree(inputs)
            right_result = right_subtree.apply_tree(inputs)
            output = tree.FUNCTIONS[function_name]["function"](
                left_result, right_result
            )
        else:
            left_subtree = tree.structure[1]
            left_subtree = Tree(left_subtree)
            left_result = left_subtree.apply_tree(inputs)
            output = tree.FUNCTIONS[function_name]["function"](left_result)
        return bound_value(output, -1000000000000.0, 10000000000000.0)
    else:  # If it's a terminal node
        if tree.structure in list(tree.TERMINALS.keys()):
            output = inputs[:, tree.TERMINALS[tree.structure]]
            return output
        elif tree.structure in list(tree.CONSTANTS.keys()):
            output = tree.CONSTANTS[tree.structure](None)
            return output


def nested_depth_calculator(operator, depths):
    """
    Calculate the depth of nested structures.

    To save computational effort, the new depth is calculated based on the operator used to generate the new tree.

    Parameters
    ----------
    operator : callable
        The operator applied to the tree.
    depths : list of int
        List of depths of subtrees.

    Returns
    -------
    int
        Maximum depth after applying the operator.
    """
    if operator.__name__ == "tt_delta_sum":
        depths[0] += 2
        depths[1] += 2
    elif operator.__name__ == "tt_delta_mul":
        depths[0] += 3
        depths[1] += 3
    elif operator.__name__ == "ot_delta_sum_True":
        depths[0] += 3
    elif operator.__name__ in ["ot_delta_sum_False", "ot_delta_mul_True"]:
        depths[0] += 4
    elif operator.__name__ == "ot_delta_mul_False":
        depths[0] += 5
    elif operator.__name__ == "geometric_crossover":
        depths = [n + 2 for n in depths]
        depths.append(depths[-1] + 1)
    return max(depths)


def nested_nodes_calculator(operator, nodes):
    """
    Calculate the number of nodes in nested structures.

    Parameters
    ----------
    operator : callable
        The operator applied to the tree.
    nodes : list of int
        List of node counts of subtrees.

    Returns
    -------
    int
        Total number of nodes after applying the operator.
    """
    extra_operators_nodes = (
        [5, nodes[-1]]
        if operator.__name__ == "geometric_crossover"
        else (
            [7]
            if operator.__name__ == "ot_delta_sum_True"
            else (
                [11]
                if operator.__name__ == "ot_delta_mul_False"
                else (
                    [9]
                    if operator.__name__ in ["ot_delta_sum_False", "ot_delta_mul_True"]
                    else (
                        [6]
                        if operator.__name__ == "tt_delta_mul"
                        else ([4] if operator.__name__ == "tt_delta_sum" else [0])
                    )
                )
            )
        )
    )
    return sum([*nodes, *extra_operators_nodes])
