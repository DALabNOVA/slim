"""
Geometric Mutation Functions for Tree Structures using PyTorch.
"""

import torch


def standard_geometric_mutation(tree, random_tree_1, random_tree_2, ms, testing, new_data = False):
    """
    Apply standard geometric mutation on tree semantics using two random trees.

    Parameters
    ----------
    tree : Tree or torch.Tensor
        The target tree whose semantics are to be mutated.
    random_tree_1 : Tree or torch.Tensor
        The first random tree for mutation.
    random_tree_2 : Tree or torch.Tensor
        The second random tree for mutation.
    ms : float
        Mutation strength.
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
    if new_data:
        return torch.add(
                tree,
                torch.mul(
                    ms,
                    torch.sub(random_tree_1, random_tree_2),
                ),
            )
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


def standard_one_tree_geometric_mutation(tree, random_tree_1, ms, testing, new_data = False):
    """
    Apply standard geometric mutation on tree semantics using one random tree.

    Parameters
    ----------
    tree : Tree or torch.Tensor
        The target tree whose semantics are to be mutated.
    random_tree_1 : Tree or torch.Tensor
        The random tree for mutation.
    ms : float
        Mutation strength.
    testing : bool
        Indicates if the operation is on test semantics.
    new_data : bool, optional
        Flag indicating whether the tree is exposed to new data outside the evolution process. If `True`,
        operations are performed on the inputs rather than semantics. Defaults to `False`.

    Returns
    -------
    torch.Tensor
        Mutated semantics or data as a torch tensor.
    """
    if new_data:
        return  torch.add(
                tree,
                torch.mul(
                    ms,
                    torch.sub(
                        1,
                        torch.div(2, torch.add(1, torch.abs(random_tree_1))),
                    ),
                ),
            )
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
        The target tree whose semantics are to be mutated.
    random_tree_1 : Tree or torch.Tensor
        The first random tree for mutation.
    random_tree_2 : Tree or torch.Tensor
        The second random tree for mutation.
    ms : float
        Mutation strength.
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

    Args:
        tree: The target tree whose semantics are to be mutated.
        random_tree_1: The random tree for mutation.
        ms: Mutation strength.
        testing: Boolean indicating if the operation is on test semantics.
    new_data : bool, optional
        Flag indicating whether the tree is exposed to new data outside the evolution process. If `True`,
        operations are performed on the inputs rather than semantics. Defaults to `False`.

    Returns:
        Mutated semantics as a torch tensor.
    """
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
