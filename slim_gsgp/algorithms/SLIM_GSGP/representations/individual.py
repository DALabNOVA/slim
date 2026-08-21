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
Individual Class and Utility Functions for SLIM GSGP.
"""

import torch
import dill
from types import SimpleNamespace
from slim_gsgp.algorithms.GSGP.representations.tree_utils import apply_tree
from slim_gsgp.utils.utils import check_slim_version

class Individual:
    """
    Individual of the SLIM_GSGP algorithm. Composed of 'blocks' of trees.

    Parameters
    ----------
    collection : list
        The list of trees representing the individual.
    structure : list
        The structure of each tree in the collection.
    size : int
        The amount of trees in the collection
    train_semantics : torch.Tensor
        Training semantics associated with the Individual.
    test_semantics : torch.Tensor or None
        Testing semantics associated with the Individual. Can be None if not applicable.
    fitness : float or None
        The fitness value of the Individual. Defaults to None.
    test_fitness : float or None
        The fitness value of the Individual during testing. Defaults to None.
    nodes_collection : int
        The number of nodes in each tree of the collection.
    nodes_count : int
        The total amount of nodes in the tree.
    depth_collection : int
        The maximum depth of each tree in the collection.
    depth : int
        The maximum depth of the tree.
    """

    def __init__(self, collection, train_semantics, test_semantics, reconstruct):
        """
        Initialize an Individual with a collection of trees and their associated semantics.

        Parameters
        ----------
        collection : list
            The list of trees representing the individual.
        train_semantics : torch.Tensor
            Training semantics associated with the individual.
        test_semantics : torch.Tensor or None
            Testing semantics associated with the individual. Can be None if not applicable.
        reconstruct : bool
            Boolean indicating if the structure of the individual should be stored.
        """
        # setting the Individual attributes based on the collection, if existent.
        # Otherwise, those are added to the individual after its created (during mutation).

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

        # setting the semantics and fitness related attributes
        self.train_semantics = train_semantics
        self.test_semantics = test_semantics
        self.fitness = None
        self.test_fitness = None

    def get_train_semantics_collapsed(self, operator, dim=0):
        return operator(self.train_semantics, dim=dim)

    def get_test_semantics_collapsed(self, operator, dim=0):
        return operator(self.test_semantics, dim=dim)

    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for the Individual. Result is stored as an attribute associated with the object.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data for calculating semantics.
        testing : bool, optional
            Boolean indicating if the calculation is for testing semantics. Default is False.

        Returns
        -------
        None
        """

        semantics_attribute = "test_semantics" if testing else "train_semantics"
        if getattr(self, semantics_attribute) is not None:
            return

        for tree in self.collection:
            tree.calculate_semantics(inputs, testing)

        semantics = [
            getattr(tree, semantics_attribute)
            for tree in self.collection
        ]
        setattr(
            self,
            semantics_attribute,
            torch.stack(
                [
                    semantic
                    if semantic.ndim != 0
                    else semantic.expand(len(inputs))
                    for semantic in semantics
                ]
            ),
        )

    def __len__(self):
        """
        Return the size of the individual.

        Returns
        -------
        int
            Size of the individual.
        """
        return self.size

    def __getitem__(self, item):
        """
        Get a tree from the individual by index.

        Parameters
        ----------
        item : int
            Index of the tree to retrieve.

        Returns
        -------
        Tree
            The tree at the specified index.
        """
        return self.collection[item]

    def evaluate(self, ffunction, y, testing=False, operator="sum"):
        """
        Evaluate the Individual using a fitness function.

        Parameters
        ----------
        ffunction : Callable
            Fitness function to evaluate the Individual.
        y : torch.Tensor
            Expected output (target) values.
        testing : bool, optional
            Boolean indicating if the evaluation is for testing semantics (default is False).
        operator : str, optional
            Operator to apply to the semantics (default is "sum").

        Returns
        -------
        None
        """
        # getting the correct torch operator based on the slim_gsgp version
        if operator == "sum":
            operator = torch.sum
        else:
            operator = torch.prod

        # computing the testing fitness, if applicable
        if testing:
            self.test_fitness = ffunction(
                y,
                torch.clamp(
                    operator(self.test_semantics, dim=0),
                    -1000000000000.0,
                    1000000000000.0,
                ),
            )
        # computing the training fitness
        else:
            self.fitness = ffunction(
                y,
                torch.clamp(
                    operator(self.train_semantics, dim=0),
                    -1000000000000.0,
                    1000000000000.0,
                ),
            )

    @staticmethod
    def _predict_block(tree, data, sig):
        """Evaluate one reconstructed SLIM block without mutating cached semantics."""
        if isinstance(tree.structure, tuple):
            return apply_tree(tree, data)

        if len(tree.structure) == 3:
            child = apply_tree(tree.structure[1], data)
            if sig:
                child = torch.sigmoid(child)
            return tree.structure[0](
                SimpleNamespace(train_semantics=child),
                tree.structure[2],
                testing=False,
            )

        left = SimpleNamespace(
            train_semantics=torch.sigmoid(apply_tree(tree.structure[1], data))
        )
        right = SimpleNamespace(
            train_semantics=torch.sigmoid(apply_tree(tree.structure[2], data))
        )
        return tree.structure[0](left, right, tree.structure[3], testing=False)

    def predict(self, data):
        """Predict from reconstructed blocks while preserving their cached semantics."""

        # seeing if the tree has the structure attribute
        if not hasattr(self, "collection"):
            raise RuntimeError("predict() is unavailable when reconstruct=False")

        # getting the relevant variables based on the used slim_gsgp version
        operator, sig, trees = check_slim_version(slim_version=self.version)

        semantics = [self._predict_block(tree, data, sig) for tree in self.collection]

        # getting the correct torch function based on the used operator (mul or sum)
        operator = torch.sum if operator == "sum" else torch.prod

        # making sure that if the semantics of the collection is solely a constant,
        # the constant value is repeated len(data) number of times to match the remaining semantics' shapes.

        semantics = [ten if ten.numel() == len(data) else ten.expand(len(data)) for ten in semantics]

        # clamping the semantics
        return torch.clamp(
            operator(torch.stack(semantics), dim=0), -1000000000000.0, 1000000000000.0
        )

    def get_tree_representation(self):
        """
        Returns a string representation of the trees in the Individual.

        Parameters
        ----------

        Returns
        -------
        str
            A string representing the structure of the trees in the individual.

        Raises
        ------
        Exception
            If reconstruct was set to False, indicating that the .get_tree_representation() method is not available.
        """
        # seeing if the tree has the structure attribute
        if not hasattr(self, "collection"):
            raise Exception("If reconstruct was set to False, .get_tree_representation() is not available")

        # finding out the used operator based on the slim_gsgp version
        operator = "sum" if "+" in self.version else "mul"

        op = "+" if operator == "sum" else "*"

        return f" {op} ".join(
            [
                str(t.structure) if isinstance(t.structure, tuple)
                else f'f({t.structure[1].structure})' if len(t.structure) == 3
                else f'f({t.structure[1].structure} - {t.structure[2].structure})'
                for t in self.collection
            ]
        )

    def print_tree_representation(self):
        """
        Prints a string representation of the trees in the Individual.

        Parameters
        ----------

        Returns
        -------
        None
            Prints a string representing the structure of the trees in the individual.
        """

        print(self.get_tree_representation())

    def save_to_file(self, file_path):
        """
        Save the Individual object to a file.

        Parameters
        ----------
        file_path : str
            The path to the file where the Individual will be saved.

        Returns
        -------
        None
        """
        with open(file_path, 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load_from_file(file_path):
        """
        Load an Individual object from a file.

        Parameters
        ----------
        file_path : str
            The path to the file from which the Individual will be loaded.

        Returns
        -------
        Individual
            The loaded Individual object.
        """
        with open(file_path, 'rb') as file:
            individual = dill.load(file)
        return individual
