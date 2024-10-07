"""
Individual Class and Utility Functions for Genetic Programming using PyTorch.
"""

import torch
from slim.algorithms.GSGP.representations.tree_utils import apply_tree
from slim.utils.utils import  check_slim_version

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
        Training semantics associated with the individual.
    test_semantics : torch.Tensor or None
        Testing semantics associated with the individual. Can be None if not applicable.
    fitness : float or None
        The fitness value of the tree. Defaults to None.
    test_fitness : float or None
        The fitness value of the tree during testing. Defaults to None.
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
        # setting the Individual attributes based on the collection, if existent. Otherwise, those are added to the individual
        # after its created (during mutation).

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

    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for the individual. Result is stored as an attribute associated with the object.

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

        # computing the testing semantics, if not existent
        if testing and self.test_semantics is None:
            # getting the semantics for every tree in the collection
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

        # computing the training semantics
        elif self.train_semantics is None:
            # getting the semantics for every tree in the collection
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
        Evaluate the individual using a fitness function.

        Parameters
        ----------
        ffunction : Callable
            Fitness function to evaluate the individual.
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

        # getting the correct torch operator based on the slim version
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

    def predict(self, data):
        """
            Predict the output for the given input data using the model's collection of trees and specified slim version.

            Parameters
            ----------
            data : array-like or DataFrame
                The input data to predict. It should be in the form of an array-like structure
                (e.g., list, numpy array) or a pandas DataFrame, where each row represents a
                different observation and each column represents a feature.

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

        # seeing if the tree has the structure attribute
        if not hasattr(self, "collection"):
            raise Exception("If reconstruct was set to False, .predict() is not available")

        # getting the relevant variables based on the used slim version
        operator, sig, trees = check_slim_version(slim_version=self.version)

        # getting an empty semantics list
        semantics = []

        # getting the semantics for each tree in the collection
        for t in self.collection:
            if isinstance(t.structure, tuple): # if it's a base (gp) tree
                semantics.append(apply_tree(t, data))
            else:
                if len(t.structure) == 3:  # one tree mutation
                    # seeing if a logistic function is to be used
                    if sig:
                        # saving the previous semantics, for safekeeping
                        t.structure[1].previous_training = t.train_semantics
                        # getting the new training semantics based on the provided data
                        t.structure[1].train_semantics = torch.sigmoid(
                            apply_tree(t.structure[1], data)
                        )
                    else:
                        # saving the previous semantics, for safekeeping
                        t.structure[1].previous_training = t.train_semantics
                        # getting the new training semantics based on the provided data
                        t.structure[1].train_semantics = apply_tree(t.structure[1], data)

                elif len(t.structure) == 4:  # two tree mutation
                    t.structure[1].previous_training = t.train_semantics
                    t.structure[1].train_semantics = torch.sigmoid(
                        apply_tree(t.structure[1], data)
                    )
                    # saving the previous semantics, for safekeeping
                    t.structure[2].previous_training = t.train_semantics
                    # getting the new training semantics based on the provided data
                    t.structure[2].train_semantics = torch.sigmoid(
                        apply_tree(t.structure[2], data)
                    )

                # getting the semantics by calling the corresponding operator on the structure
                semantics.append(t.structure[0](*t.structure[1:], testing=False))

        # getting the correct torch function based on the used operator (mul or sum)
        operator = torch.sum if operator == "sum" else torch.prod

        # clamping the semantics
        return torch.clamp(
            operator(torch.stack(semantics), dim=0), -1000000000000.0, 1000000000000.0
        )

    def get_tree_representation(self, operator=None):
        """
        Get a string representation of the trees in the individual.

        Parameters
        ----------
        operator : str, optional
            The operator to use in the representation ("sum" or "mul"). If None, it will be determined based on the version.

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

        # finding out the used operator based on the slim version
        if operator is None:
            operator = "sum" if "+" in self.version else "mul"

        op = "+" if operator == "sum" else "*"

        return f" {op} ".join([str(t.structure) if isinstance(t.structure,
                                                              tuple) else f'f({t.structure[1].structure})' if len(
            t.structure) == 3
        else f'f({t.structure[1].structure} - {t.structure[2].structure})' for t in self.collection])
