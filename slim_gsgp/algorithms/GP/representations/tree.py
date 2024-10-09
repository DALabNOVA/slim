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
Tree class implementation for representing tree structures in genetic programming.
"""

from slim_gsgp.algorithms.GP.representations.tree_utils import bound_value, flatten, tree_depth, _execute_tree
import torch

class Tree:
    """
    The Tree class representing the candidate solutions in genetic programming.

    Attributes
    ----------
    repr_ : tuple or str
        Representation of the tree structure.
    FUNCTIONS : dict
        Dictionary of allowed functions in the tree representation.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree representation.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree representation.
    depth : int
        Depth of the tree.
    fitness : float
        Fitness value of the tree.
    test_fitness : float
        Test fitness value of the tree.
    node_count : int
        Number of nodes in the tree.

    Methods
    -------
    __init__(repr_)
        Initializes a Tree object.
    apply_tree(inputs)
        Evaluates the tree on input vectors.
    evaluate(ffunction, X, y, testing=False)
        Evaluates the tree given a fitness function and data.
    predict(X):
        Outputs a prediction for a given input X.
    print_tree_representation(indent="")
        Prints the tree representation with indentation.
    """

    TERMINALS = None
    FUNCTIONS = None
    CONSTANTS = None

    def __init__(self, repr_):
        """
        Initializes a Tree object.

        Parameters
        ----------
        repr_ : tuple
            Representation of the tree structure.
        """
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS

        self.repr_ = repr_
        self.depth = tree_depth(Tree.FUNCTIONS)(repr_)
        self.fitness = None
        self.test_fitness = None
        self.node_count = len(list(flatten(self.repr_)))

    def apply_tree(self, inputs):
        """
        Evaluates the tree on input vectors.

        Parameters
        ----------
        inputs : tuple
            Input vectors.

        Returns
        -------
        float
            Output of the evaluated tree.
        """

        return _execute_tree(
            repr_=self.repr_,
            X=inputs,
            FUNCTIONS=self.FUNCTIONS,
            TERMINALS=self.TERMINALS,
            CONSTANTS=self.CONSTANTS
        )

    def evaluate(self, ffunction, X, y, testing=False, new_data = False):
        """
        Evaluates the tree given a fitness function, input data (X), and target data (y).

        Parameters
        ----------
        ffunction : function
            Fitness function to evaluate the individual.
        X : torch.Tensor
            The input data (which can be training or testing).
        y : torch.Tensor
            The expected output (target) values.
        testing : bool, optional
            Flag indicating if the data is testing data. Default is False.
        new_data : bool, optional
            Flag indicating that the input data is new and the model is being used outside the training process.

        Returns
        -------
        None
            If the data is training or testing data, the fitness value is attributed to the individual.
        float
            If exposed to new data, the fitness value is returned.
        """
        # getting the predictions (i.e., semantics) of the individual
        preds = self.apply_tree(X)

        # if new (testing data) is being used, return the fitness value for this new data
        if new_data:
            return float(ffunction(y, preds))

        # if not, attribute the fitness value to the individual
        else:
            if testing:
                self.test_fitness = ffunction(y, preds)
            else:
                self.fitness = ffunction(y, preds)

    def predict(self, X):
        """
        Predict the tree semantics (output) for the given input data.

        Parameters
        ----------
        X : torch.Tensor
            The input data to predict.

        Returns
        -------
        torch.Tensor
            The predicted output for the input data.

        Notes
        -----
        This function delegates the actual prediction task to the `apply_tree` method,
        which is assumed to be another method in the same class. The `apply_tree` method
        should be defined to handle the specifics of how predictions are made based on
        the tree structure used in this model.
        """
        return self.apply_tree(X)

    def print_tree_representation(self, indent=""):
        """
        Prints the tree representation with indentation.

        Parameters
        ----------
        indent : str, optional
            Indentation for tree structure representation. Default is an empty string.
        """
        if isinstance(self.repr_, tuple):  # If it's a function node
            function_name = self.repr_[0]
            print(indent + f"{function_name}(")
            # if the function has an arity of 2, print both left and right subtrees
            if Tree.FUNCTIONS[function_name]["arity"] == 2:
                left_subtree, right_subtree = self.repr_[1], self.repr_[2]
                Tree(left_subtree).print_tree_representation(indent + "  ")
                Tree(right_subtree).print_tree_representation(indent + "  ")
            # if the function has an arity of 1, print the left subtree
            else:
                left_subtree = self.repr_[1]
                Tree(left_subtree).print_tree_representation(indent + "  ")
            print(indent + ")")
        else:  # If it's a terminal node
            print(indent + f"{self.repr_}")
