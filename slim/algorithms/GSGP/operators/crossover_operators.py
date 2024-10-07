"""
Geometric crossover implementation for genetic programming trees.
"""

import torch


def geometric_crossover(tree1, tree2, random_tree, testing, new_data=False):
    """
    Performs geometric crossover between two trees using a random tree.

    Parameters
    ----------
    tree1 : Tree or torch.Tensor
        The first parent tree. If geometric_crossover is called with new_data=True, it means the final tree is being
        evaluated on testing data and tree1 is a torch.Tensor. Otherwise, during training, the individuals are Tree instances.

    tree2 : Tree or torch.Tensor
        The second parent tree. If geometric_crossover is called with new_data=True, it means the final tree is being
        evaluated on testing data and tree2 is a torch.Tensor. Otherwise, during training, the individuals are Tree instances.

    random_tree : Tree or torch.Tensor
        The random tree used for crossover. If geometric_crossover is called with new_data=True, it means the final tree is being
        evaluated on testing data and random_tree is a torch.Tensor. Otherwise, during training, random_tree is a Tree instance.
    testing : bool
        Flag indicating whether to use test semantics or train semantics.
    new_data : bool
        Flag indicating whether the trees are exposed to new data, outside the evolution process. In this case,
        operations are performed on the inputs rather than semantics.
    Returns
    -------
    torch.Tensor
        The semantics of the individual, resulting from geometric crossover.
    """
    # if new (testing) data is used (for the testing of the final tree), return the semantics resulting from crossover
    if new_data:
        return torch.add(
            torch.mul(tree1, random_tree),
            torch.mul(torch.sub(1, random_tree), tree2),
        )
    # if new_data is false, geomettric_crossover is being called during GSGP's training phase, tree.test_semantics or tree.train_semantics
    # attribute is used
    else:
        if testing:
            return torch.add(
                torch.mul(tree1.test_semantics, random_tree.test_semantics),
                torch.mul(torch.sub(1, random_tree.test_semantics), tree2.test_semantics),
            )
        else:
            return torch.add(
                torch.mul(tree1.train_semantics, random_tree.train_semantics),
                torch.mul(torch.sub(1, random_tree.train_semantics), tree2.train_semantics),
            )
