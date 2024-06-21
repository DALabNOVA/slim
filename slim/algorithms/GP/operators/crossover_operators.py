"""
Crossover operator implementation.
"""

from slim.algorithms.GP.representations.tree_utils import (random_subtree,
                                                           substitute_subtree)


def crossover_trees(FUNCTIONS):
    """
    Performs crossover between two trees.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of allowed functions in the trees.

    Returns
    -------
    function
        Function to perform crossover between two trees.
    """
    subtree_substitution = substitute_subtree(FUNCTIONS=FUNCTIONS)
    random_subtree_picker = random_subtree(FUNCTIONS=FUNCTIONS)

    def inner_xo(tree1, tree2, tree1_n_nodes, tree2_n_nodes):
        """
        Inner function to perform crossover between two trees.

        Parameters
        ----------
        tree1 : tuple
            The first tree.
        tree2 : tuple
            The second tree.
        tree1_n_nodes : int
            Number of nodes in the first tree.
        tree2_n_nodes : int
            Number of nodes in the second tree.

        Returns
        -------
        tuple
            Two new trees after performing crossover.
        """
        if isinstance(tree1, tuple) and isinstance(tree2, tuple):
            # Randomly select crossover points in both trees
            crossover_point_tree1 = random_subtree_picker(
                tree1, num_of_nodes=tree1_n_nodes
            )
            crossover_point_tree2 = random_subtree_picker(
                tree2, num_of_nodes=tree2_n_nodes
            )

            # Swap subtrees at the crossover points
            new_tree1 = subtree_substitution(
                tree1, crossover_point_tree1, crossover_point_tree2
            )
            new_tree2 = subtree_substitution(
                tree2, crossover_point_tree2, crossover_point_tree1
            )

            return new_tree1, new_tree2
        else:
            # If either tree is a terminal node, return them as they are (no crossover)
            return tree1, tree2

    return inner_xo
