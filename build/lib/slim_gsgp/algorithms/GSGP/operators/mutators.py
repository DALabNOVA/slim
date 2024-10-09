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
Geometric Mutation Functions for Tree Structures using PyTorch.
"""

import torch


def standard_geometric_mutation(tree, random_tree_1, random_tree_2, ms, testing, new_data=False):
    """
    Apply standard geometric mutation on tree semantics using two random trees.

    Parameters
    ----------
    tree : Tree or torch.Tensor
        The target tree whose semantics are to be mutated. If standard_geometric_mutation is called with new_data=True,
        it means the final tree is being evaluated on testing data and tree is a torch.Tensor. Otherwise,
        during training, the individuals are Tree instances.
    random_tree_1 : Tree or torch.Tensor
        The first random tree for mutation. If standard_geometric_mutation is called with new_data=True, it means the
        final tree is being evaluated on testing data and random_tree_1 is a torch.Tensor. Otherwise, during training,
        random_tree_1 is a Tree instance.
    random_tree_2 : Tree or torch.Tensor
        The second random tree for mutation. If standard_geometric_mutation is called with new_data=True, it means the
        final tree is being evaluated on testing data and random_tree_2 is a torch.Tensor. Otherwise, during training,
        random_tree_2 is a Tree instance.
    ms : float
        Mutation step.
    testing : bool
        Indicates if the operation is on test semantics.
    new_data : bool, optional
        Flag indicating whether the trees are exposed to new data outside the evolution process. If `True`,
        operations are performed on the inputs rather than semantics. Defaults to `False`.

    Returns
    -------
    torch.Tensor
        Mutated semantics or data as a torch tensor.
    """
    # if new (testing) data is used (for the testing of the final tree), return the semantics resulting from mutation
    if new_data:
        return torch.add(
                tree,
                torch.mul(
                    ms,
                    torch.sub(random_tree_1, random_tree_2),
                ),
            )
    # if new_data is false, standard_geometric_mutation is being called during GSGP's training phase,
    # tree.test_semantics or tree.train_semantics attribute is used
    else:
        if testing:
            return torch.add(
                tree.test_semantics,
                torch.mul(
                    ms,
                    torch.sub(random_tree_1.test_semantics, random_tree_2.test_semantics),
                ),
            )
        else:
            return torch.add(
                tree.train_semantics,
                torch.mul(
                    ms,
                    torch.sub(random_tree_1.train_semantics, random_tree_2.train_semantics),
                ),
            )


def standard_one_tree_geometric_mutation(tree, random_tree_1, ms, testing, new_data=False):
    """
    Apply standard geometric mutation on tree semantics using one random tree.

    Parameters
    ----------
    tree : Tree or torch.Tensor
        The target tree whose semantics are to be mutated. If standard_one_tree_geometric_mutation is called with
        new_data=True, it means the final tree is being evaluated on testing data and tree is a torch.Tensor.
        Otherwise, during training, the individuals are Tree instances.
    random_tree_1 : Tree or torch.Tensor
        The random tree for mutation. If standard_one_tree_geometric_mutation is called with new_data=True,
        it means the final tree is being evaluated on testing data and random_tree_1 is a torch.Tensor.
        Otherwise, during training, random_tree_1 is a Tree instance.
    ms : float
        Mutation step.
    testing : bool
        Indicates if the operation is on test semantics.
    new_data : bool, optional
        Flag indicating whether the tree is exposed to new data outside the evolution process. If `True`,
        operations are performed on the inputs rather than semantics. Defaults to `False`.

    Returns
    -------
    torch.Tensor
        Mutated semantics of the individual.
    """
    # if new (testing) data is used (for the testing of the final tree), return the semantics resulting from mutation
    if new_data:
        return torch.add(
                tree,
                torch.mul(
                    ms,
                    torch.sub(
                        1,
                        torch.div(2, torch.add(1, torch.abs(random_tree_1))),
                    ),
                ),
            )
    # if new_data is false, standard_geometric_mutation is being called during GSGP's training phase,
    # tree.test_semantics or tree.train_semantics attribute is used
    else:
        if testing:
            return torch.add(
                tree.test_semantics,
                torch.mul(
                    ms,
                    torch.sub(
                        1,
                        torch.div(2, torch.add(1, torch.abs(random_tree_1.test_semantics))),
                    ),
                ),
            )
        else:
            return torch.add(
                tree.train_semantics,
                torch.mul(
                    ms,
                    torch.sub(
                        1,
                        torch.div(
                            2, torch.add(1, torch.abs(random_tree_1.train_semantics))
                        ),
                    ),
                ),
            )


def product_two_trees_geometric_mutation(tree, random_tree_1, random_tree_2, ms, testing, new_data = False):
    """
    Apply product-based geometric mutation on tree semantics using two random trees.

    Parameters
    ----------
    tree : Tree or torch.Tensor
        The target tree whose semantics are to be mutated. If product_two_trees_geometric_mutation is called with
        new_data=True, it means the final tree is being evaluated on testing data and tree is a torch.Tensor.
        Otherwise, during training, the individuals are Tree instances.
    random_tree_1 : Tree or torch.Tensor
        The first random tree for mutation.  If product_two_trees_geometric_mutation is called with new_data=True,
        it means the final tree is being evaluated on testing data and random_tree_1 is a torch.Tensor.
        Otherwise, during training, random_tree_1 is a Tree instance.
    random_tree_2 : Tree or torch.Tensor
        The second random tree for mutation. If product_two_trees_geometric_mutation is called with new_data=True,
        it means the final tree is being evaluated on testing data and random_tree_2 is a torch.Tensor.
        Otherwise, during training, random_tree_2 is a Tree instance.
    ms : float
        Mutation step.
    testing : bool
        Indicates if the operation is on test semantics.
    new_data : bool, optional
        Flag indicating whether the tree is exposed to new data outside the evolution process. If `True`,
        operations are performed on the inputs rather than semantics. Defaults to `False`.

    Returns
    -------
    torch.Tensor
        Mutated semantics as a torch tensor.
    """
    # if new (testing) data is used (for the testing of the final tree), return the semantics resulting from mutation
    if new_data:
        return torch.mul(
                tree,
                torch.add(
                    1,
                    torch.mul(
                        ms,
                        torch.sub(
                            random_tree_1, random_tree_2
                        ),
                    ),
                ),
            )
    # if new_data is false, standard_geometric_mutation is being called during GSGP's training phase,
    # tree.test_semantics or tree.train_semantics attribute is used
    else:
        if testing:
            return torch.mul(
                tree.test_semantics,
                torch.add(
                    1,
                    torch.mul(
                        ms,
                        torch.sub(
                            random_tree_1.test_semantics, random_tree_2.test_semantics
                        ),
                    ),
                ),
            )
        else:
            return torch.mul(
                tree.train_semantics,
                torch.add(
                    1,
                    torch.mul(
                        ms,
                        torch.sub(
                            random_tree_1.train_semantics, random_tree_2.train_semantics
                        ),
                    ),
                ),
            )


def product_one_trees_geometric_mutation(tree, random_tree_1, ms, testing, new_data = False):
    """
        Apply product-based geometric mutation on tree semantics using one random tree.

        Parameters
        ----------
        tree : Tree or torch.Tensor
            The target tree whose semantics are to be mutated. If product_one_trees_geometric_mutation is called with
            new_data=True, it means the final tree is being evaluated on testing data and tree is a torch.Tensor.
            Otherwise, during training, the individuals are Tree instances.
        random_tree_1 : Tree or torch.Tensor
            The random tree for mutation. If product_one_trees_geometric_mutation is called with new_data=True,
            it means the final tree is being evaluated on testing data and random_tree_1 is a torch.Tensor.
            Otherwise, during training, random_tree_1 is a Tree instance.
        ms : float
            Mutation step.
        testing : bool
            Boolean indicating if the operation is on test semantics.
        new_data : bool, optional
            Flag indicating whether the tree is exposed to new data outside the evolution process. If True,
            operations are performed on the inputs rather than semantics. Defaults to False.

        Returns
        -------
        torch.Tensor
            Mutated semantics as a torch tensor.
        """
    # if new (testing) data is used (for the testing of the final tree), return the semantics resulting from mutation
    if new_data:
        return torch.mul(
            tree,
            torch.add(
                1,
                torch.mul(
                    ms,
                    torch.sub(
                        1,
                        torch.div(
                            2, torch.add(1, torch.abs(random_tree_1))
                        ),
                    ),
                ),
            ),
        )
    # if new_data is false, standard_geometric_mutation is being called during GSGP's training phase,
    # tree.test_semantics or tree.train_semantics attribute is used
    else:
        if testing:
            return torch.mul(
                tree.test_semantics,
                torch.add(
                    1,
                    torch.mul(
                        ms,
                        torch.sub(
                            1,
                            torch.div(
                                2, torch.add(1, torch.abs(random_tree_1.test_semantics))
                            ),
                        ),
                    ),
                ),
            )
        else:
            return torch.mul(
                tree.train_semantics,
                torch.add(
                    1,
                    torch.mul(
                        ms,
                        torch.sub(
                            1,
                            torch.div(
                                2, torch.add(1, torch.abs(random_tree_1.train_semantics))
                            ),
                        ),
                    ),
                ),
            )
