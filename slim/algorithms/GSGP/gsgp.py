"""
Genetic Programming (GP) and Geometric Semantic Genetic Programming (GSGP) modules.
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

        Args:
            pi_init (dict): Dictionary with all the parameters needed for evaluation.
            initializer (function): Function to initialize the population.
            selector (function): Function to select individuals for crossover/mutation.
            mutator (function): Function to mutate individuals.
            ms (function): Function to determine mutation step.
            crossover (function): Function to perform crossover between individuals.
            find_elit_func (function): Function to find elite individuals.
            p_m (float): Probability of mutation.
            p_xo (float): Probability of crossover.
            pop_size (int): Size of the population.
            seed (int): Seed for random number generation.
            settings_dict (dict): Additional settings dictionary.
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
    ):
        """
        Execute the GSGP algorithm.

        Args:
            x_train (torch.Tensor): Training data features.
            x_test (torch.Tensor): Test data features.
            y_train (torch.Tensor): Training data labels.
            y_test (torch.Tensor): Test data labels.
            curr_dataset (str): Current dataset name.
            n_iter (int): Number of iterations.
            elitism (bool): Whether to use elitism.
            log (int): Logging level.
            verbose (int): Verbosity level.
            test_elite (bool): Whether to test elite individuals.
            log_path (str): Path to save logs.
            run_info (list): Information about the current run.
            ffunction (function): Fitness function.
            reconstruct (bool): Whether to reconstruct trees.
            n_elites (int): Number of elites.
        """
        if test_elite and (X_test is None or y_test is None):
            raise Exception('If test_elite is True you need to provide a test dataset')

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

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

        population.calculate_semantics(X_train)
        if test_elite:
            population.calculate_semantics(X_test, testing=True)
        population.evaluate(ffunction, y=y_train)

        end = time.time()
        self.elites, self.elite = self.find_elit_func(population, n_elites)
        if test_elite:
            self.elite.evaluate(ffunction, y=y_test, testing=True)

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

        if verbose != 0:
            verbose_reporter(
                curr_dataset,
                0,
                self.elite.fitness,
                self.elite.test_fitness,
                end - start,
                self.elite.nodes,
            )

        for it in range(1, n_iter + 1, 1):
            offs_pop, start = [], time.time()
            if elitism:
                offs_pop.append(self.elite)

            while len(offs_pop) < self.pop_size:

                if random.random() < self.p_xo:
                    p1, p2 = self.selector(population), self.selector(population)

                    while p1 == p2:
                        p1, p2 = self.selector(population), self.selector(population)

                    r_tree = get_random_tree(
                        max_depth=self.pi_init["init_depth"],
                        FUNCTIONS=self.pi_init["FUNCTIONS"],
                        TERMINALS=self.pi_init["TERMINALS"],
                        CONSTANTS=self.pi_init["CONSTANTS"],
                        inputs=X_train,
                        logistic=True,
                        p_c=self.pi_init["p_c"],
                    )

                    if test_elite:
                        r_tree.calculate_semantics(X_test, testing=True, logistic=True)

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
                    if not reconstruct:
                        offs1.nodes = nested_nodes_calculator(
                            self.crossover, [p1.nodes, p2.nodes, r_tree.nodes]
                        )
                        offs1.depth = nested_depth_calculator(
                            self.crossover, [p1.depth, p2.depth, r_tree.depth]
                        )

                    offs_pop.append(offs1)

                else:
                    p1 = self.selector(population)
                    ms_ = self.ms()

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

                        mutation_trees = [r_tree1, r_tree2]

                        if test_elite:
                            [
                                rt.calculate_semantics(
                                    X_test, testing=True, logistic=True
                                )
                                for rt in mutation_trees
                            ]

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

                        mutation_trees = [r_tree1]

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

                    offs_pop.append(offs1)
                    if not reconstruct:
                        offs1.nodes = nested_nodes_calculator(
                            self.mutator,
                            [p1.nodes, *[rt.nodes for rt in mutation_trees]],
                        )
                        offs1.depth = nested_depth_calculator(
                            self.mutator,
                            [p1.depth, *[rt.depth for rt in mutation_trees]],
                        )

            if len(offs_pop) > population.size:
                offs_pop = offs_pop[: population.size]

            offs_pop = Population(offs_pop)
            offs_pop.evaluate(ffunction, y=y_train)
            population = offs_pop

            end = time.time()

            self.elites, self.elite = self.find_elit_func(population, n_elites)

            if test_elite:
                self.elite.evaluate(ffunction, y=y_test, testing=True)

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

            if verbose != 0:
                verbose_reporter(
                    run_info[-1],
                    it,
                    self.elite.fitness,
                    self.elite.test_fitness,
                    end - start,
                    self.elite.nodes,
                )
