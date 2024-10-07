"""
Crossover operator implementation.
"""

from slim.algorithms.GP.representations.tree_utils import (random_subtree,
                                                           substitute_subtree)


def crossover_trees(FUNCTIONS):
    """
    Returns a function that performs crossover between two tree representations.

    To avoid passing the FUNCTIONS parameter unnecessarily, a new function is created utilizing it. This function is
    returned and passed as a parameter to the GP algorithm, where it is then called when crossover is performed.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of allowed functions in the trees.

    Returns
    -------
    Callable
        A function (`inner_xo`) that performs crossover between two tree representations.
        Inner function to perform crossover between two trees.

        Parameters
        ----------
        tree1 : tuple
            The first tree representation.
        tree2 : tuple
            The second tree representation.
        tree1_n_nodes : int
            Number of nodes in the first tree representation.
        tree2_n_nodes : int
            Number of nodes in the second tree representation.

        Returns
        -------
        tuple
            Two new tree representations after performing crossover.
        Notes
        -----
        This function selects random crossover points from both `tree1` and `tree2` and swaps
        their subtrees at those points. If either tree is a terminal node, it returns the tree
        representations unchanged.

    Notes
    -----
    The returned function (`inner_xo`) takes two tree representations and their node counts,
    selects random subtrees, and swaps them to create the representations of the new offspring trees.
    """
    # getting the function to substitute a subtree in a tree
    subtree_substitution = substitute_subtree(FUNCTIONS=FUNCTIONS)
    # getting the random subtree selection function
    random_subtree_picker = random_subtree(FUNCTIONS=FUNCTIONS)

    def inner_xo(tree1, tree2, tree1_n_nodes, tree2_n_nodes):
        """
        Inner function to perform crossover between two trees.

        Parameters
        ----------
        tree1 : tuple
            The first tree representation.
        tree2 : tuple
            The second tree representation.
        tree1_n_nodes : int
            Number of nodes in the first tree representation.
        tree2_n_nodes : int
            Number of nodes in the second tree representation.

        Returns
        -------
        tuple
            Two new trees after performing crossover.
        Notes
        -----
        This function selects random crossover points from both `tree1` and `tree2` and swaps
        their subtrees at those points. If either tree is a terminal node, it returns the trees
        representations unchanged.
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
            # If either tree1 or tree2 is a terminal node, return them as they are (no crossover)
            return tree1, tree2

    return inner_xo
