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
Mutation Functions for SLIM GSGP.
"""

import random

import torch
from slim.algorithms.GSGP.representations.tree import Tree
from slim.algorithms.SLIM_GSGP.representations.individual import Individual
from slim.utils.utils import get_random_tree


# two tree function
def two_trees_delta(operator="sum"):
    """
    Generate a function for the two-tree delta mutation.

    Parameters
    ----------
    operator : str
        The operator to be used in the mutation ("sum" or other).

    Returns
    -------
    Callable
        A mutation function (`tt_delta`) for two Individuals that returns the mutated semantics.

        Parameters
        ----------
        tr1 : Individual
            The first tree individual.
        tr2 : Individual
            The second tree individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train Individual semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.

    Notes
    -----
    The returned function ('tt_delta_{operator}') takes as input two individuals, the mutation step, a boolean
    representing whether to use the train or test semantics, and returns the calculated semantics of the new individual.
    """

    def tt_delta(tr1, tr2, ms, testing):
        """
        Performs delta mutation between two trees based on their semantics.

        Parameters
        ----------
        tr1 : Individual
            The first tree Individual.
        tr2 : Individual
            The second tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train Individual semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.
        """
        if testing:
            return (
                torch.mul(ms, torch.sub(tr1.test_semantics, tr2.test_semantics))
                if operator == "sum"
                else torch.add(
                    1, torch.mul(ms, torch.sub(tr1.test_semantics, tr2.test_semantics))
                )
            )

        else:
            return (
                torch.mul(ms, torch.sub(tr1.train_semantics, tr2.train_semantics))
                if operator == "sum"
                else torch.add(
                    1,
                    torch.mul(ms, torch.sub(tr1.train_semantics, tr2.train_semantics)),
                )
            )

    tt_delta.__name__ += "_" + operator

    return tt_delta


def one_tree_delta(operator="sum", sig=False):
    """
    Generate a function for the one-tree delta mutation.

    Parameters
    ----------
    operator : str
        The operator to be used in the mutation ("sum" or other).
    sig : bool
        Boolean indicating if sigmoid should be applied.

    Returns
    -------
    Callable
        A mutation function (`ot_delta`) for one-tree mutation.

        Parameters
        ----------
        tr1 : Individual
            The tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.
    Notes
    -----
    The returned function ('ot_delta_{operator}_{sig}') takes as input one individual, the mutation step,
    a boolean representing whether to use the train or test semantics, and returns the mutated semantics.
    """
    def ot_delta(tr1, ms, testing):
        """
        Performs delta mutation on one tree based on its semantics.

        Parameters
        ----------
        tr1 : Individual
            The tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.
        """
        if sig:
            if testing:
                return (
                    torch.mul(ms, torch.sub(torch.mul(2, tr1.test_semantics), 1))
                    if operator == "sum"
                    else torch.add(
                        1, torch.mul(ms, torch.sub(torch.mul(2, tr1.test_semantics), 1))
                    )
                )
            else:
                return (
                    torch.mul(ms, torch.sub(torch.mul(2, tr1.train_semantics), 1))
                    if operator == "sum"
                    else torch.add(
                        1,
                        torch.mul(ms, torch.sub(torch.mul(2, tr1.train_semantics), 1)),
                    )
                )
        else:
            if testing:
                return (
                    torch.mul(
                        ms,
                        torch.sub(
                            1, torch.div(2, torch.add(1, torch.abs(tr1.test_semantics)))
                        ),
                    )
                    if operator == "sum"
                    else torch.add(
                        1,
                        torch.mul(
                            ms,
                            torch.sub(
                                1,
                                torch.div(
                                    2, torch.add(1, torch.abs(tr1.test_semantics))
                                ),
                            ),
                        ),
                    )
                )
            else:
                return (
                    torch.mul(
                        ms,
                        torch.sub(
                            1,
                            torch.div(2, torch.add(1, torch.abs(tr1.train_semantics))),
                        ),
                    )
                    if operator == "sum"
                    else torch.add(
                        1,
                        torch.mul(
                            ms,
                            torch.sub(
                                1,
                                torch.div(
                                    2, torch.add(1, torch.abs(tr1.train_semantics))
                                ),
                            ),
                        ),
                    )
                )

    ot_delta.__name__ += "_" + operator + "_" + str(sig)
    return ot_delta


def inflate_mutation(FUNCTIONS, TERMINALS,CONSTANTS,two_trees=True,operator="sum",single_tree_sigmoid=False,sig=False):
    """
    Generate an inflate mutation function.

    Parameters
    ----------
    FUNCTIONS : dict
        The dictionary of functions used in the mutation.
    TERMINALS : dict
        The dictionary of terminals used in the mutation.
    CONSTANTS : dict
        The dictionary of constants used in the mutation.
    two_trees : bool
        Boolean indicating if two trees should be used.
    operator : str
        The operator to be used in the mutation.
    single_tree_sigmoid : bool
        Boolean indicating if sigmoid should be applied to a single tree.
    sig : bool
        Boolean indicating if sigmoid should be applied.

    Returns
    -------
    Callable
        An inflate mutation function (`inflate`).

        Parameters
        ----------
        individual : Individual
            The tree Individual to mutate.
        ms : float
            Mutation step.
        X : torch.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        X_test : torch.Tensor, optional
            Test data for calculating test semantics (default: None).
        grow_probability : float, optional
            Probability of growing trees during mutation (default: 1).
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after mutation (default: True).

        Returns
        -------
        Individual
            The mutated tree Individual.

    Notes
    -----
    The returned function performs inflate mutation on Individuals, using either one or two randomly generated trees
    and applying either delta mutation or sigmoid mutation based on the parameters.
    """
    def inflate(
        individual,
        ms,
        X,
        max_depth=8,
        p_c=0.1,
        X_test=None,
        grow_probability=1,
        reconstruct=True,
    ):
        """
        Perform inflate mutation on the given Individual.

        Parameters
        ----------
        individual : Individual
            The tree Individual to mutate.
        ms : float
            Mutation step.
        X : torch.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        X_test : torch.Tensor, optional
            Test data for calculating test semantics (default: None).
        grow_probability : float, optional
            Probability of growing trees during mutation (default: 1).
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after mutation (default: True).

        Returns
        -------
        Individual
            The mutated tree Individual.
        """
        if two_trees:
            # getting two random trees
            random_tree1 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=True,
            )
            random_tree2 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=True,
            )
            # adding the random trees to a list, to be used in the creation of a new block
            random_trees = [random_tree1, random_tree2]

            # calculating the semantics of the random trees on testing, if applicable
            if X_test is not None:
                [
                    rt.calculate_semantics(X_test, testing=True, logistic=True)
                    for rt in random_trees
                ]

        else:
            # getting one random tree
            random_tree1 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=single_tree_sigmoid or sig,
            )
            # adding the random tree to a list, to be used in the creation of a new block
            random_trees = [random_tree1]

            # calculating the semantics of the random trees on testing, if applicable
            if X_test is not None:
                [
                    rt.calculate_semantics(
                        X_test, testing=True, logistic=single_tree_sigmoid or sig
                    )
                    for rt in random_trees
                ]

        # getting the correct mutation operator, based on the number of random trees used
        variator = (
            two_trees_delta(operator=operator)
            if two_trees
            else one_tree_delta(operator=operator, sig=sig)
        )
        # creating the new block for the individual, based on the random trees and operators
        new_block = Tree(
            structure=[variator, *random_trees, ms],
            train_semantics=variator(*random_trees, ms, testing=False),
            test_semantics=(
                variator(*random_trees, ms, testing=True)
                if X_test is not None
                else None
            ),
            reconstruct=True,
        )
        # creating the offspring individual, by adding the new block to it
        offs = Individual(
            collection=[*individual.collection, new_block] if reconstruct else None,
            train_semantics=torch.stack(
                [
                    *individual.train_semantics,
                    (
                        new_block.train_semantics
                        if new_block.train_semantics.shape != torch.Size([])
                        else new_block.train_semantics.repeat(len(X))
                    ),
                ]
            ),
            test_semantics=(
                (
                    torch.stack(
                        [
                            *individual.test_semantics,
                            (
                                new_block.test_semantics
                                if new_block.test_semantics.shape != torch.Size([])
                                else new_block.test_semantics.repeat(len(X_test))
                            ),
                        ]
                    )
                )
                if individual.test_semantics is not None
                else None
            ),
            reconstruct=reconstruct,
        )
        # computing offspring attributes
        offs.size = individual.size + 1
        offs.nodes_collection = [*individual.nodes_collection, new_block.nodes]
        offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

        offs.depth_collection = [*individual.depth_collection, new_block.depth]
        offs.depth = max(
            [
                depth - (i - 1) if i != 0 else depth
                for i, depth in enumerate(offs.depth_collection)
            ]
        ) + (offs.size - 1)

        return offs

    return inflate


def deflate_mutation(individual, reconstruct):
    """
    Perform deflate mutation on a given Individual by removing a random 'block'.

    Parameters
    ----------
    individual : Individual
        The Individual to be mutated.
    reconstruct : bool
        Whether to store the Individual's structure after mutation.

    Returns
    -------
    Individual
        The mutated individual
    """
    # choosing the block that will be removed
    mut_point = random.randint(1, individual.size - 1)

    # removing the block from the individual and creating a new Individual
    offs = Individual(
        collection=(
            [
                *individual.collection[:mut_point],
                *individual.collection[mut_point + 1 :],
            ]
            if reconstruct
            else None
        ),
        train_semantics=torch.stack(
            [
                *individual.train_semantics[:mut_point],
                *individual.train_semantics[mut_point + 1 :],
            ]
        ),
        test_semantics=(
            torch.stack(
                [
                    *individual.test_semantics[:mut_point],
                    *individual.test_semantics[mut_point + 1 :],
                ]
            )
            if individual.test_semantics is not None
            else None
        ),
        reconstruct=reconstruct,
    )

    # computing offspring attributes
    offs.size = individual.size - 1
    offs.nodes_collection = [
        *individual.nodes_collection[:mut_point],
        *individual.nodes_collection[mut_point + 1 :],
    ]
    offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

    offs.depth_collection = [
        *individual.depth_collection[:mut_point],
        *individual.depth_collection[mut_point + 1 :],
    ]
    offs.depth = max(
        [
            depth - (i - 1) if i != 0 else depth
            for i, depth in enumerate(offs.depth_collection)
        ]
    ) + (offs.size - 1)

    return offs
