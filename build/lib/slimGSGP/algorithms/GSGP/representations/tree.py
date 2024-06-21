"""
Tree Class for Genetic Programming using PyTorch.
"""

import torch
from slim.algorithms.GP.representations.tree_utils import flatten, tree_depth
from slim.algorithms.GSGP.representations.tree_utils import (
    apply_tree, nested_depth_calculator, nested_nodes_calculator)
from slim.algorithms.GP.representations.tree import Tree as GP_Tree


class Tree:
    FUNCTIONS = None
    TERMINALS = None
    CONSTANTS = None

    def __init__(self, structure, train_semantics, test_semantics, reconstruct):
        """
        Initialize the Tree with its structure and semantics.

        Args:
            structure: The tree structure, either as a tuple or a list of pointers.
            train_semantics: The training semantics associated with the tree.
            test_semantics: The testing semantics associated with the tree.
            reconstruct: Boolean indicating if the tree should be reconstructed.
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

        Args:
            inputs: Input data for calculating semantics.
            testing: Boolean indicating if the calculation is for testing semantics.
            logistic: Boolean indicating if a logistic function should be applied.

        Returns:
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

    def evaluate(self, ffunction, y, testing=False, X = None):
        """
        Evaluate the tree using a fitness function.

        Args:
            ffunction: Fitness function to evaluate the individual.
            y: Expected output (target) values as a torch tensor.
            testing: Boolean indicating if the evaluation is for testing semantics.

        Returns:
            None
        """
        if X is not None:
            semantics = apply_tree(self, X) if isinstance(self.structure, tuple) \
                else self.structure[0](*self.structure[1:], testing=False)
            ffunction(y, semantics)
        else:
            if testing:
                self.test_fitness = ffunction(y, self.test_semantics)
            else:
                self.fitness = ffunction(y, self.train_semantics)

    def predict(self, data):
        """
            Predict the output for the given input data using the model's structure.

            Parameters
            ----------
            data : array-like or DataFrame
                The input data to predict. It should be in the form of an array-like structure
                (e.g., list, numpy array) or a pandas DataFrame, where each row represents a
                different observation and each column represents a feature.

            Returns
            -------
            array-like
                The predicted output for the input data. The exact form and type of the output
                depend on whether the model's structure is a tuple or a list, as well as the
                specific implementation details of the `apply_tree` function and the prediction
                methods of the tree objects within the structure.

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
        if isinstance(self.structure, tuple):
            return apply_tree(self, data)
        else:
            ms = [ms for ms in self.structure[1:] if isinstance(ms, float)]
            base_trees = list(filter(lambda x: isinstance(x, Tree), self.structure))
            return self.structure[0](*[tree.predict(data) for tree in base_trees], *ms, testing = False, new_data = True)


