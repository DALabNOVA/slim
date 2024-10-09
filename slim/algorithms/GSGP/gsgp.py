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
Geometric Semantic Genetic Programming (GSGP) module.
"""

import random
import time

import numpy as np
import torch
from slim.algorithms.GP.representations.tree import Tree as GP_Tree
from slim.algorithms.GSGP.representations.population import Population
from slim.algorithms.GSGP.representations.tree import Tree
from slim.algorithms.GSGP.representations.tree_utils import (
    nested_depth_calculator, nested_nodes_calculator)
from slim.utils.diversity import gsgp_pop_div_from_vectors
from slim.utils.logger import logger
from slim.utils.utils import get_random_tree, verbose_reporter


class GSGP:
    """
    Geometric Semantic Genetic Programming class.
    """

    def __init__(
        self,
        pi_init,
        initializer,
        selector,
        mutator,
        ms,
        crossover,
        find_elit_func,
        p_m=0.8,
        p_xo=0.2,
        pop_size=100,
        seed=0,
        settings_dict=None,
    ):
        """
        Initialize the GSGP algorithm.

        Parameters
        ----------
        pi_init : dict
            Dictionary with all the parameters needed for candidate solutions initialization.
        initializer : callable
            Function to initialize the population.
        selector : callable
            Function to select individuals.
        mutator : callable
            Function to mutate individuals.
        ms : callable
            Function to determine mutation step.
        crossover : callable
            Function to perform crossover between individuals.
        find_elit_func : callable
            Function to find elite individuals.
        p_m : float, optional
            Probability of mutation. Defaults to 0.8.
        p_xo : float, optional
            Probability of crossover. Defaults to 0.2.
        pop_size : int, optional
            Size of the population. Defaults to 100.
        seed : int, optional
            Seed for random number generation. Defaults to 0.
        settings_dict : dict, optional
            Additional settings dictionary. Defaults to None.
        """

        self.pi_init = pi_init
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.mutator = mutator
        self.ms = ms
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed
        self.settings_dict = settings_dict
        self.find_elit_func = find_elit_func

        Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        Tree.TERMINALS = pi_init["TERMINALS"]
        Tree.CONSTANTS = pi_init["CONSTANTS"]
        GP_Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        GP_Tree.TERMINALS = pi_init["TERMINALS"]
        GP_Tree.CONSTANTS = pi_init["CONSTANTS"]

    def solve(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        curr_dataset,
        n_iter=20,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        run_info=None,
        ffunction=None,
        reconstruct=False,
        n_elites=1,
        n_jobs=1
    ):
        """
        Execute the GSGP algorithm.

        Parameters
        ----------
        X_train : torch.Tensor
            Training data features.
        X_test : torch.Tensor
            Test data features.
        y_train : torch.Tensor
            Training data labels.
        y_test : torch.Tensor
            Test data labels.
        curr_dataset : str
            Current dataset name.
        n_iter : int
            Number of iterations.
        elitism : bool
            Whether to use elitism.
        log : int
            Logging level. Default is 0.
        verbose : int
            Verbosity level. Default is 0.
        test_elite : bool
            Whether to evaluate elite individuals on test data. Default is False.
        log_path : str
            Path to save logs. Default is None.
        run_info : list
            Information about the current run. Default is None.
        ffunction : callable
            Fitness function. Default is None.
        reconstruct : bool
            Whether to reconstruct trees. Default is False.
        n_elites : int
            Number of elites. Default is 1.
        n_jobs : int
            The maximum number of jobs for parallel processing. Default is 1.
        """

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # starting the time computation
        start = time.time()

        # creating the initial population
        population = Population(
            [
                Tree(
                    structure=tree,
                    train_semantics=None,
                    test_semantics=None,
                    reconstruct=True,
                )
                for tree in self.initializer(**self.pi_init)
            ]
        )

        # calculating the semantics for the initial population
        population.calculate_semantics(X_train)

        # calculating the testing semantics, if applicable
        if test_elite:
            population.calculate_semantics(X_test, testing=True)

        # evaluating the initial population
        population.evaluate(ffunction, y=y_train, n_jobs=n_jobs)

        end = time.time()

        # obtaining the elite(s)
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        # testing the elite, if applicable
        if test_elite:
            self.elite.evaluate(ffunction, y=y_test, testing=True)

        # logging the intial results, if the result level is != 0
        if log != 0:
            if log == 2:
                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes,
                    gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                (
                                    ind.train_semantics
                                    if ind.train_semantics.shape != torch.Size([])
                                    else ind.train_semantics.repeat(len(X_train))
                                )
                                for ind in population.population
                            ]
                        )
                    ),
                    np.std(population.fit),
                    log,
                ]

            elif log == 3:

                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes,
                    " ".join([str(ind.nodes_count) for ind in population.population]),
                    " ".join([str(f) for f in population.fit]),
                    log,
                ]

            elif log == 4:

                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes,
                    gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                (
                                    ind.train_semantics
                                    if ind.train_semantics.shape != torch.Size([])
                                    else ind.train_semantics.repeat(len(X_train))
                                )
                                for ind in population.population
                            ]
                        )
                    ),
                    np.std(population.fit),
                    " ".join([str(ind.nodes) for ind in population.population]),
                    " ".join([str(f) for f in population.fit]),
                    log,
                ]

            else:

                add_info = [self.elite.test_fitness, self.elite.nodes, log]

            # logging the results
            logger(
                log_path,
                0,
                self.elite.fitness,
                end - start,
                float(population.nodes_count),
                additional_infos=add_info,
                run_info=run_info,
                seed=self.seed,
            )
        # displaying the results on console, if applicable
        if verbose != 0:
            verbose_reporter(
                curr_dataset,
                0,
                self.elite.fitness,
                self.elite.test_fitness,
                end - start,
                self.elite.nodes,
            )
        # EVOLUTIONARY PROCESS
        for it in range(1, n_iter + 1, 1):
            # creating the empty offpsring population
            offs_pop, start = [], time.time()
            # adding the elite(s) to the offspring population
            if elitism:
                offs_pop.extend(self.elites)

            # filling the offspring population
            while len(offs_pop) < self.pop_size:

                # if choosing xover
                if random.random() < self.p_xo:
                    p1, p2 = self.selector(population), self.selector(population)

                    # assuring the parents are different
                    while p1 == p2:
                        p1, p2 = self.selector(population), self.selector(population)
                    # getting the random tree
                    r_tree = get_random_tree(
                        max_depth=self.pi_init["init_depth"],
                        FUNCTIONS=self.pi_init["FUNCTIONS"],
                        TERMINALS=self.pi_init["TERMINALS"],
                        CONSTANTS=self.pi_init["CONSTANTS"],
                        inputs=X_train,
                        logistic=True,
                        p_c=self.pi_init["p_c"],
                    )
                    # getting the testing semantics of the random trees, if applicable
                    if test_elite:
                        r_tree.calculate_semantics(X_test, testing=True, logistic=True)

                    # getting the offspring
                    offs1 = Tree(
                        structure=(
                            [self.crossover, p1, p2, r_tree] if reconstruct else None
                        ),
                        train_semantics=self.crossover(p1, p2, r_tree, testing=False),
                        test_semantics=(
                            self.crossover(p1, p2, r_tree, testing=True)
                            if test_elite
                            else None
                        ),
                        reconstruct=reconstruct,
                    )

                    # if reconstruct = False, calculate the nodes and the depth of the offspring
                    if not reconstruct:
                        offs1.nodes = nested_nodes_calculator(
                            self.crossover, [p1.nodes, p2.nodes, r_tree.nodes]
                        )
                        offs1.depth = nested_depth_calculator(
                            self.crossover, [p1.depth, p2.depth, r_tree.depth]
                        )

                    # adding the offspring to the offspring population
                    offs_pop.append(offs1)

                # if mutation
                else:
                    # selecting the parent
                    p1 = self.selector(population)

                    # determining the mutation step
                    ms_ = self.ms()

                    # seeing if two random trees are needed or only one
                    if self.mutator.__name__ in [
                        "standard_geometric_mutation",
                        "product_two_trees_geometric_mutation",
                    ]:

                        r_tree1 = get_random_tree(
                            max_depth=self.pi_init["init_depth"],
                            FUNCTIONS=self.pi_init["FUNCTIONS"],
                            TERMINALS=self.pi_init["TERMINALS"],
                            CONSTANTS=self.pi_init["CONSTANTS"],
                            inputs=X_train,
                            p_c=self.pi_init["p_c"],
                        )

                        r_tree2 = get_random_tree(
                            max_depth=self.pi_init["init_depth"],
                            FUNCTIONS=self.pi_init["FUNCTIONS"],
                            TERMINALS=self.pi_init["TERMINALS"],
                            CONSTANTS=self.pi_init["CONSTANTS"],
                            inputs=X_train,
                            p_c=self.pi_init["p_c"],
                        )

                        # storing the mutation trees
                        mutation_trees = [r_tree1, r_tree2]

                        # calculating the testing semantics of the random tree(s), if applicable
                        if test_elite:
                            [
                                rt.calculate_semantics(
                                    X_test, testing=True, logistic=True
                                )
                                for rt in mutation_trees
                            ]

                    # getting only one random tree
                    else:
                        r_tree1 = get_random_tree(
                            max_depth=self.pi_init["init_depth"],
                            FUNCTIONS=self.pi_init["FUNCTIONS"],
                            TERMINALS=self.pi_init["TERMINALS"],
                            CONSTANTS=self.pi_init["CONSTANTS"],
                            inputs=X_train,
                            logistic=False,
                            p_c=self.pi_init["p_c"],
                        )

                        # storing the mutation tree
                        mutation_trees = [r_tree1]

                        # calculating the testing semantics of the random tree, if applicable
                        if test_elite:
                            r_tree1.calculate_semantics(
                                X_test, testing=True, logistic=False
                            )

                    offs1 = Tree(
                        structure=(
                            [self.mutator, p1, *mutation_trees, ms_]
                            if reconstruct
                            else None
                        ),
                        train_semantics=self.mutator(
                            p1, *mutation_trees, ms_, testing=False
                        ),
                        test_semantics=(
                            self.mutator(p1, *mutation_trees, ms_, testing=True)
                            if test_elite
                            else None
                        ),
                        reconstruct=reconstruct,
                    )

                    # adding the offspring to the population
                    offs_pop.append(offs1)

                    # if reconstruct = False, calculate the nodes and the depth of the offspring

                    if not reconstruct:
                        offs1.nodes = nested_nodes_calculator(
                            self.mutator,
                            [p1.nodes, *[rt.nodes for rt in mutation_trees]],
                        )
                        offs1.depth = nested_depth_calculator(
                            self.mutator,
                            [p1.depth, *[rt.depth for rt in mutation_trees]],
                        )

            # making sure the offspring population contains population.size amount of individuals
            if len(offs_pop) > population.size:
                offs_pop = offs_pop[: population.size]

            # turning the offspring population into a Population
            offs_pop = Population(offs_pop)

            # evaluating the offspring population
            offs_pop.evaluate(ffunction, y=y_train, n_jobs=n_jobs)

            # replacing the population with the offspring population (P = P')
            population = offs_pop

            end = time.time()

            # replacing the elites
            self.elites, self.elite = self.find_elit_func(population, n_elites)

            # testing elite, if applicable
            if test_elite:
                self.elite.evaluate(ffunction, y=y_test, testing=True)

            # logging the results if log !=0
            if log != 0:

                if log == 2:
                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes,
                        gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    (
                                        ind.train_semantics
                                        if ind.train_semantics.shape != torch.Size([])
                                        else ind.train_semantics.repeat(len(X_train))
                                    )
                                    for ind in population.population
                                ]
                            )
                        ),
                        np.std(population.fit),
                        log,
                    ]

                elif log == 3:

                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes,
                        " ".join(
                            [str(ind.nodes_count) for ind in population.population]
                        ),
                        " ".join([str(f) for f in population.fit]),
                        log,
                    ]

                elif log == 4:

                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes,
                        gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    (
                                        ind.train_semantics
                                        if ind.train_semantics.shape != torch.Size([])
                                        else ind.train_semantics.repeat(len(X_train))
                                    )
                                    for ind in population.population
                                ]
                            )
                        ),
                        np.std(population.fit),
                        " ".join([str(ind.nodes) for ind in population.population]),
                        " ".join([str(f) for f in population.fit]),
                        log,
                    ]

                else:

                    add_info = [self.elite.test_fitness, self.elite.nodes, log]

                # logging the results
                logger(
                    log_path,
                    it,
                    self.elite.fitness,
                    end - start,
                    float(population.nodes_count),
                    additional_infos=add_info,
                    run_info=run_info,
                    seed=self.seed,
                )
            # displaying the results on console, if applicable
            if verbose != 0:
                verbose_reporter(
                    run_info[-1],
                    it,
                    self.elite.fitness,
                    self.elite.test_fitness,
                    end - start,
                    self.elite.nodes,
                )
