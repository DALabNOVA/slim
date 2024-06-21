"""
Individual Class and Utility Functions for Genetic Programming using PyTorch.
"""

import torch
from slim.algorithms.GSGP.representations.tree_utils import apply_tree
from slim.utils.utils import  check_slim_version


class Individual:
    """
    Initialize an Individual with a collection of trees and semantics.

    Args:
        collection: List of trees representing the individual.
        train_semantics: Training semantics associated with the individual.
        test_semantics: Testing semantics associated with the individual.
        reconstruct: Boolean indicating if the individual should be reconstructed.
    """

    def __init__(self, collection, train_semantics, test_semantics, reconstruct):
        if collection is not None and reconstruct:
            self.collection = collection
            self.structure = [tree.structure for tree in collection]
            self.size = len(collection)

            self.nodes_collection = [tree.nodes for tree in collection]
            self.nodes_count = sum(self.nodes_collection) + (self.size - 1)
            self.depth_collection = [tree.depth for tree in collection]
            self.depth = max(
                [
                    depth - (i - 1) if i != 0 else depth
                    for i, depth in enumerate(self.depth_collection)
                ]
            ) + (self.size - 1)

        self.train_semantics = train_semantics
        self.test_semantics = test_semantics
        self.fitness = None
        self.test_fitness = None

    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for the individual.

        Args:
            inputs: Input data for calculating semantics.
            testing: Boolean indicating if the calculation is for testing semantics.

        Returns:
            None
        """

        if testing and self.test_semantics is None:
            [tree.calculate_semantics(inputs, testing) for tree in self.collection]
            self.test_semantics = torch.stack(
                [
                    (
                        tree.test_semantics
                        if tree.test_semantics.shape != torch.Size([])
                        else tree.test_semantics.repeat(len(inputs))
                    )
                    for tree in self.collection
                ]
            )

        elif self.train_semantics is None:
            [tree.calculate_semantics(inputs, testing) for tree in self.collection]
            self.train_semantics = torch.stack(
                [
                    (
                        tree.train_semantics
                        if tree.train_semantics.shape != torch.Size([])
                        else tree.train_semantics.repeat(len(inputs))
                    )
                    for tree in self.collection
                ]
            )

    def __len__(self):
        """
        Return the size of the individual.

        Returns:
            int: Size of the individual.
        """
        return self.size

    def __getitem__(self, item):
        """Get a tree from the individual by index.

        Args:
            item: Index of the tree to retrieve.

        Returns:
            Tree: The tree at the specified index.
        """
        return self.collection[item]

    def evaluate(self, ffunction, y, testing=False, operator="sum"):
        """
        Evaluate the individual using a fitness function.

        Args:
            ffunction: Fitness function to evaluate the individual.
            y: Expected output (target) values as a torch tensor.
            testing: Boolean indicating if the evaluation is for testing semantics.
            operator: Operator to apply to the semantics ("sum" or "prod").

        Returns:
            None
        """
        if operator == "sum":
            operator = torch.sum
        else:
            operator = torch.prod

        if testing:
            self.test_fitness = ffunction(
                y,
                torch.clamp(
                    operator(self.test_semantics, dim=0),
                    -1000000000000.0,
                    1000000000000.0,
                ),
            )

        else:
            self.fitness = ffunction(
                y,
                torch.clamp(
                    operator(self.train_semantics, dim=0),
                    -1000000000000.0,
                    1000000000000.0,
                ),
            )


    def predict(self, data, slim_version):
        """
            Predict the output for the given input data using the model's collection of trees and specified slim version.

            Parameters
            ----------
            data : array-like or DataFrame
                The input data to predict. It should be in the form of an array-like structure
                (e.g., list, numpy array) or a pandas DataFrame, where each row represents a
                different observation and each column represents a feature.

            slim_version : bool
                A flag to indicate whether to use a slim version of the model for prediction.
                The exact meaning of slim version is determined by the `check_slim_version` function.

            Returns
            -------
            Tensor
                The predicted output for the input data. The output is a tensor whose values
                are clamped between -1e12 and 1e12.

            Notes
            -----
            The prediction process involves several steps:

            1. The `check_slim_version` function is called with the `slim_version` flag to determine
               the appropriate operator (`sum` or `prod`), whether to apply a sigmoid function (`sig`),
               and the specific trees to use for prediction.

            2. For each tree in the `self.collection`:
               - If the tree structure is a tuple, predictions are made using the `apply_tree` function.
               - If the tree structure is a list:
                 - For single-tree structures (length 3), predictions are made directly or with a sigmoid
                   function applied, and training semantics are updated.
                 - For two-tree structures (length 4), predictions for both trees are made with a sigmoid
                   function applied, and training semantics are updated for both trees.

            3. The semantics (predicted outputs) of all trees are combined using the specified operator
               (`sum` or `prod`), and the final output is clamped to be within the range of -1e12 to 1e12.

            This function relies on PyTorch for tensor operations, including `torch.sigmoid`,
            `torch.sum`, `torch.prod`, `torch.stack`, and `torch.clamp`.
            """
        operator, sig, trees = check_slim_version(slim_version=slim_version)

        semantics = []

        for t in self.collection:
            if isinstance(t.structure, tuple):
                semantics.append(apply_tree(t, data))
            else:

                if len(t.structure) == 3:  # one tree
                    if sig:
                        t.structure[1].previous_training = t.train_semantics
                        t.structure[1].train_semantics = torch.sigmoid(
                            apply_tree(t.structure[1], data)
                        )
                    else:
                        t.structure[1].previous_training = t.train_semantics
                        t.structure[1].train_semantics = apply_tree(t.structure[1], data)

                elif len(t.structure) == 4:  # two tree
                    t.structure[1].previous_training = t.train_semantics
                    t.structure[1].train_semantics = torch.sigmoid(
                        apply_tree(t.structure[1], data)
                    )

                    t.structure[2].previous_training = t.train_semantics
                    t.structure[2].train_semantics = torch.sigmoid(
                        apply_tree(t.structure[2], data)
                    )

                semantics.append(t.structure[0](*t.structure[1:], testing=False))

        operator = torch.sum if operator == "sum" else torch.prod

        return torch.clamp(
            operator(torch.stack(semantics), dim=0), -1000000000000.0, 1000000000000.0
        )
