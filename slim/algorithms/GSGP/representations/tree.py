"""
Tree Class for Genetic Programming using PyTorch.
"""

import torch
from slim.algorithms.GP.representations.tree_utils import flatten, tree_depth
from slim.algorithms.GSGP.representations.tree_utils import apply_tree, nested_depth_calculator, nested_nodes_calculator
from slim.algorithms.GSGP.operators.crossover_operators import geometric_crossover


class Tree:
    FUNCTIONS = None
    TERMINALS = None
    CONSTANTS = None

    def __init__(self, structure, train_semantics, test_semantics, reconstruct):
        """
        Initialize the Tree with its structure and semantics.

        Parameters
        ----------
        structure : tuple or list
            The tree structure, either as a tuple or a list of pointers.
        train_semantics : torch.Tensor
            The training semantics associated with the tree.
        test_semantics : torch.Tensor
            The testing semantics associated with the tree.
        reconstruct : bool
            Indicates if the tree's structure should be stored for later reconstruction.

        Attributes
        ----------
        depth : int
            The maximum depth of the tree structure.
        nodes : int
            The total number of nodes in the tree.
        fitness : float or None
            The fitness value of the tree. Defaults to None.
        test_fitness : float or None
            The fitness value of the tree during testing. Defaults to None.
        """
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS

        if structure is not None and reconstruct:
            self.structure = (
                structure  # either repr_ from gp(tuple) or list of pointers
            )

        self.train_semantics = train_semantics
        self.test_semantics = test_semantics

        if isinstance(structure, tuple):
            self.depth = tree_depth(Tree.FUNCTIONS)(structure)
            self.nodes = len(list(flatten(structure)))
        elif reconstruct:
            self.depth = nested_depth_calculator(
                self.structure[0],
                [tree.depth for tree in self.structure[1:] if isinstance(tree, Tree)],
            )
            self.nodes = nested_nodes_calculator(
                self.structure[0],
                [tree.nodes for tree in self.structure[1:] if isinstance(tree, Tree)],
            )

        self.fitness = None
        self.test_fitness = None

    def calculate_semantics(self, inputs, testing=False, logistic=False):
        """
        Calculate the semantics for the tree.

        Semantics are stored as an attribute in their respective objects.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data for calculating semantics.
        testing : bool, optional
            Indicates if the calculation is for testing semantics. Defaults to `False`.
        logistic : bool, optional
            Indicates if a logistic function should be applied. Defaults to `False`.

        Returns
        -------
        None
        """
        if testing and self.test_semantics is None:
            if isinstance(self.structure, tuple):
                self.test_semantics = (
                    torch.sigmoid(apply_tree(self, inputs))
                    if logistic
                    else apply_tree(self, inputs)
                )
            else:
                self.test_semantics = self.structure[0](
                    *self.structure[1:], testing=True
                )
        elif self.train_semantics is None:
            if isinstance(self.structure, tuple):
                self.train_semantics = (
                    torch.sigmoid(apply_tree(self, inputs))
                    if logistic
                    else apply_tree(self, inputs)
                )
            else:
                self.train_semantics = self.structure[0](
                    *self.structure[1:], testing=False
                )

    def evaluate(self, ffunction, y, testing=False, X=None):
        """
        Evaluate the tree using a fitness function.

        During the evolution process, stores the fitness as an attribute. If evaluating with new data, fitness is
        returned as a float.

        Parameters
        ----------
        ffunction : callable
            Fitness function to evaluate the individual.
        y : torch.Tensor
            Expected output (target) values as a torch tensor.
        testing : bool, optional
            Indicates if the evaluation is for testing semantics. Defaults to `False`.
        X : array-like, optional
            Input data used for calculation. Optional inside the evolution process as only the semantics are needed,
            but necessary outside of it.

        Returns
        -------
        None or float
            Returns nothing or the fitness result.
        """
        if X is not None:
            semantics = apply_tree(self, X) if isinstance(self.structure, tuple) \
                else self.structure[0](*self.structure[1:], testing=False)
            return ffunction(y, semantics)
        else:
            if testing:
                self.test_fitness = ffunction(y, self.test_semantics)
            else:
                self.fitness = ffunction(y, self.train_semantics)


    def predict(self, data):
        """
        Predict the output for the given input data using the model's structure.

        Uses recursive logic to call itself on the structure of the tree until arriving at a basic tuple structure, and
        then applies the necessary operations to arrive at the final result for the whole tree.

        Parameters
        ----------
        data : torch.Tensor
            The input data to predict.

        Returns
        -------
        torch.Tensor
            The predicted output for the input data.

        Notes
        -----
        The prediction process depends on the structure of the model:

        - If `self.structure` is a tuple, the `apply_tree` function is used for prediction.
        - If `self.structure` is a list, the first element is assumed to be a function that
          combines the predictions of multiple base trees (contained in the list) along with
          additional parameters (floats) extracted from the list. The base trees are instances
          of the `Tree` class, and their individual predictions are passed to the combining
          function along with any extracted parameters.

        The combining function is called with the predictions of the base trees and the
        extracted parameters, along with `testing` set to False and `new_data` set to True.
        """

        # seeing if the tree has the structure attribute
        if not hasattr(self, "structure"):
            raise Exception("If reconstruct was set to False, .predict() is not available.")

        if isinstance(self.structure, tuple):
            return apply_tree(self, data)
        else:
            ms = [ms for ms in self.structure[1:] if isinstance(ms, float)]
            base_trees = list(filter(lambda x: isinstance(x, Tree), self.structure))

            if self.structure[0] == geometric_crossover:
                return self.structure[0](*[tree.predict(data) for tree in base_trees[:-1]], torch.sigmoid(base_trees[-1].predict(data)), testing=False, new_data=True)
            else:
                # only apply the sigmoid to the random trees (in indexes 1 and 2)
                return self.structure[0](*[torch.sigmoid(tree.predict(data))  if i != 0 else tree.predict(data) for i, tree in enumerate(base_trees) ], *ms, testing = False, new_data = True)



