"""
Mutation Functions for Genetic Programming using PyTorch.
"""

import random

import torch
from slim.algorithms.GSGP.representations.tree import Tree
from slim.algorithms.SLIM_GSGP.representations.individual import Individual
from slim.utils.utils import get_random_tree


def two_trees_delta(operator="sum"):
    """
    Generate a function for two-tree delta mutation.

    Args:
        operator: The operator to be used in the mutation ("sum" or other).

    Returns:
        Function: A mutation function for two trees.
    """

    def tt_delta(tr1, tr2, ms, testing):
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
    Generate a function for one-tree delta mutation.

    Args:
        operator: The operator to be used in the mutation ("sum" or other).
        sig: Boolean indicating if sigmoid should be applied.

    Returns:
        Function: A mutation function for one tree.
    """

    def ot_delta(tr1, ms, testing):
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


def inflate_mutation(
    FUNCTIONS,
    TERMINALS,
    CONSTANTS,
    two_trees=True,
    operator="sum",
    single_tree_sigmoid=False,
    sig=False,
):
    """
    Generate an inflate mutation function.

    Args:
        FUNCTIONS: List of functions used in the mutation.
        TERMINALS: List of terminals used in the mutation.
        CONSTANTS: List of constants used in the mutation.
        two_trees: Boolean indicating if two trees should be used.
        operator: The operator to be used in the mutation.
        single_tree_sigmoid: Boolean indicating if sigmoid should be applied to single tree.
        sig: Boolean indicating if sigmoid should be applied.

    Returns:
        Function: An inflate mutation function.
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
            random_trees = [random_tree1, random_tree2]

            # getting the testing semantics of the random trees
            if X_test is not None:
                [
                    rt.calculate_semantics(X_test, testing=True, logistic=True)
                    for rt in random_trees
                ]

        else:
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

            random_trees = [random_tree1]

            if X_test is not None:
                [
                    rt.calculate_semantics(
                        X_test, testing=True, logistic=single_tree_sigmoid or sig
                    )
                    for rt in random_trees
                ]

        variator = (
            two_trees_delta(operator=operator)
            if two_trees
            else one_tree_delta(operator=operator, sig=sig)
        )
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
    Perform deflate mutation on an individual.

    Args:
        individual: The individual to be mutated.
        reconstruct: Boolean indicating if the individual should be reconstructed.

    Returns:
        Individual: The mutated individual.
    """
    mut_point = random.randint(1, individual.size - 1)
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
