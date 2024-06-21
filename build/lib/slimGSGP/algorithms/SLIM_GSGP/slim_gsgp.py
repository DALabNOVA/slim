"""
SLIM_GSGP Class for Evolutionary Computation using PyTorch.
"""

import random
import time

import numpy as np
import torch
from slim.algorithms.GP.representations.tree import Tree as GP_Tree
from slim.algorithms.GSGP.representations.tree import Tree
from slim.algorithms.SLIM_GSGP.representations.individual import Individual
from slim.algorithms.SLIM_GSGP.representations.population import Population
from slim.utils.diversity import gsgp_pop_div_from_vectors
from slim.utils.logger import logger
from slim.utils.utils import verbose_reporter


class SLIM_GSGP:

    def __init__(
        self,
        pi_init,
        initializer,
        selector,
        inflate_mutator,
        deflate_mutator,
        ms,
        crossover,
        find_elit_func,
        p_m=1,
        p_xo=0,
        p_inflate=0.3,
        p_deflate=0.7,
        pop_size=100,
        seed=0,
        operator="sum",
        copy_parent=True,
        two_trees=True,
        settings_dict=None,
    ):
        """
        Initialize the SLIM_GSGP algorithm with given parameters.

        Args:
            pi_init: Dictionary with all the parameters needed for evaluation.
            initializer: Function to initialize the population.
            selector: Function to select individuals from the population.
            inflate_mutator: Function for inflate mutation.
            deflate_mutator: Function for deflate mutation.
            ms: Mutation step function.
            crossover: Crossover function.
            find_elit_func: Function to find elite individuals.
            p_m: Probability of mutation.
            p_xo: Probability of crossover.
            p_inflate: Probability of inflate mutation.
            p_deflate: Probability of deflate mutation.
            pop_size: Population size.
            seed: Random seed.
            operator: Operator to apply to the semantics ("sum" or "prod").
            copy_parent: Boolean indicating if parent should be copied when mutation is not possible.
            two_trees: Boolean indicating if two trees are used.
            settings_dict: Additional settings dictionary.
        """

        self.pi_init = pi_init
        self.selector = selector
        self.p_m = p_m
        self.p_inflate = p_inflate
        self.p_deflate = p_deflate
        self.crossover = crossover
        self.inflate_mutator = inflate_mutator
        self.deflate_mutator = deflate_mutator
        self.ms = ms
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed
        self.operator = operator
        self.copy_parent = copy_parent
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
        run_info,
        n_iter=20,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        ffunction=None,
        max_depth=17,
        n_elites=1,
        reconstruct=True,
    ):
        """
        Solve the optimization problem using SLIM_GSGP.

        Args:
            X_train: Training input data.
            X_test: Testing input data.
            y_train: Training output data.
            y_test: Testing output data.
            curr_dataset: Current dataset identifier.
            run_info: Information about the current run.
            n_iter: Number of iterations.
            elitism: Boolean indicating if elitism is used.
            log: Logging level.
            verbose: Verbosity level.
            test_elite: Boolean indicating if elite should be tested.
            log_path: Path for logging.
            ffunction: Fitness function.
            max_depth: Maximum depth for trees.
            n_elites: Number of elite individuals.
            reconstruct: Boolean indicating if reconstruction is needed.
        """

        if test_elite and (X_test is None or y_test is None):
            raise Exception('If test_elite is True you need to provide a test dataset')

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        population = Population(
            [
                Individual(
                    collection=[
                        Tree(
                            tree,
                            train_semantics=None,
                            test_semantics=None,
                            reconstruct=True,
                        )
                    ],
                    train_semantics=None,
                    test_semantics=None,
                    reconstruct=True,
                )
                for tree in self.initializer(**self.pi_init)
            ]
        )

        population.calculate_semantics(X_train)
        population.evaluate(ffunction, y=y_train, operator=self.operator)

        end = time.time()

        self.elites, self.elite = self.find_elit_func(population, n_elites)

        if test_elite:
            population.calculate_semantics(X_test, testing=True)
            self.elite.evaluate(
                ffunction, y=y_test, testing=True, operator=self.operator
            )

        if log != 0:
            if log == 2:
                gen_diversity = (
                    gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.sum(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        ),
                    )
                    if self.operator == "sum"
                    else gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.prod(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        )
                    )
                )
                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes_count,
                    float(gen_diversity),
                    np.std(population.fit),
                    log,
                ]

            elif log == 3:
                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes_count,
                    " ".join([str(ind.nodes_count) for ind in population.population]),
                    " ".join([str(f) for f in population.fit]),
                    log,
                ]

            elif log == 4:
                gen_diversity = (
                    gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.sum(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        ),
                    )
                    if self.operator == "sum"
                    else gsgp_pop_div_from_vectors(
                        torch.stack(
                            [
                                torch.prod(ind.train_semantics, dim=0)
                                for ind in population.population
                            ]
                        )
                    )
                )
                add_info = [
                    self.elite.test_fitness,
                    self.elite.nodes_count,
                    float(gen_diversity),
                    np.std(population.fit),
                    " ".join([str(ind.nodes_count) for ind in population.population]),
                    " ".join([str(f) for f in population.fit]),
                    log,
                ]

            else:

                add_info = [self.elite.test_fitness, self.elite.nodes_count, log]

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
                self.elite.nodes_count,
            )

        for it in range(1, n_iter + 1, 1):
            offs_pop, start = [], time.time()
            if elitism:
                offs_pop.extend(self.elites)
            while len(offs_pop) < self.pop_size:
                if random.random() < self.p_xo:
                    p1, p2 = self.selector(population), self.selector(population)
                    while p1 == p2:
                        p1, p2 = self.selector(population), self.selector(population)
                    pass  # implement crossover
                else:
                    if random.random() < self.p_deflate:
                        p1 = self.selector(population, deflate=False)
                        if p1.size == 1:
                            if self.copy_parent:
                                off1 = Individual(
                                    collection=p1.collection if reconstruct else None,
                                    train_semantics=p1.train_semantics,
                                    test_semantics=p1.test_semantics,
                                    reconstruct=reconstruct,
                                )
                                (
                                    off1.nodes_collection,
                                    off1.nodes_count,
                                    off1.depth_collection,
                                    off1.depth,
                                    off1.size,
                                ) = (
                                    p1.nodes_collection,
                                    p1.nodes_count,
                                    p1.depth_collection,
                                    p1.depth,
                                    p1.size,
                                )
                            else:
                                ms_ = self.ms()
                                off1 = self.inflate_mutator(
                                    p1,
                                    ms_,
                                    X_train,
                                    max_depth=self.pi_init["init_depth"],
                                    p_c=self.pi_init["p_c"],
                                    X_test=X_test,
                                    reconstruct=reconstruct,
                                )

                        else:
                            off1 = self.deflate_mutator(p1, reconstruct=reconstruct)

                    else:
                        p1 = self.selector(population, deflate=False)
                        ms_ = self.ms()
                        if max_depth is not None and p1.depth == max_depth:
                            if self.copy_parent:
                                off1 = Individual(
                                    collection=p1.collection if reconstruct else None,
                                    train_semantics=p1.train_semantics,
                                    test_semantics=p1.test_semantics,
                                    reconstruct=reconstruct,
                                )
                                (
                                    off1.nodes_collection,
                                    off1.nodes_count,
                                    off1.depth_collection,
                                    off1.depth,
                                    off1.size,
                                ) = (
                                    p1.nodes_collection,
                                    p1.nodes_count,
                                    p1.depth_collection,
                                    p1.depth,
                                    p1.size,
                                )
                            else:
                                off1 = self.deflate_mutator(p1, reconstruct=reconstruct)

                        else:

                            off1 = self.inflate_mutator(
                                p1,
                                ms_,
                                X_train,
                                max_depth=self.pi_init["init_depth"],
                                p_c=self.pi_init["p_c"],
                                X_test=X_test,
                                reconstruct=reconstruct,
                            )

                        if max_depth is not None and off1.depth > max_depth:
                            if self.copy_parent:
                                off1 = Individual(
                                    collection=p1.collection if reconstruct else None,
                                    train_semantics=p1.train_semantics,
                                    test_semantics=p1.test_semantics,
                                    reconstruct=reconstruct,
                                )
                                (
                                    off1.nodes_collection,
                                    off1.nodes_count,
                                    off1.depth_collection,
                                    off1.depth,
                                    off1.size,
                                ) = (
                                    p1.nodes_collection,
                                    p1.nodes_count,
                                    p1.depth_collection,
                                    p1.depth,
                                    p1.size,
                                )
                            else:
                                off1 = self.deflate_mutator(p1, reconstruct=reconstruct)

                    offs_pop.append(off1)

            if len(offs_pop) > population.size:

                offs_pop = offs_pop[: population.size]

            offs_pop = Population(offs_pop)
            offs_pop.calculate_semantics(X_train)

            offs_pop.evaluate(ffunction, y=y_train, operator=self.operator)
            population = offs_pop

            self.population = population

            end = time.time()

            self.elites, self.elite = self.find_elit_func(population, n_elites)

            if test_elite:
                self.elite.calculate_semantics(X_test, testing=True)
                self.elite.evaluate(
                    ffunction, y=y_test, testing=True, operator=self.operator
                )

            if log != 0:

                if log == 2:
                    gen_diversity = (
                        gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    torch.sum(ind.train_semantics, dim=0)
                                    for ind in population.population
                                ]
                            ),
                        )
                        if self.operator == "sum"
                        else gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    torch.prod(ind.train_semantics, dim=0)
                                    for ind in population.population
                                ]
                            )
                        )
                    )
                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes_count,
                        float(gen_diversity),
                        np.std(population.fit),
                        log,
                    ]

                elif log == 3:
                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes_count,
                        " ".join(
                            [str(ind.nodes_count) for ind in population.population]
                        ),
                        " ".join([str(f) for f in population.fit]),
                        log,
                    ]

                elif log == 4:
                    gen_diversity = (
                        gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    torch.sum(ind.train_semantics, dim=0)
                                    for ind in population.population
                                ]
                            ),
                        )
                        if self.operator == "sum"
                        else gsgp_pop_div_from_vectors(
                            torch.stack(
                                [
                                    torch.prod(ind.train_semantics, dim=0)
                                    for ind in population.population
                                ]
                            )
                        )
                    )
                    add_info = [
                        self.elite.test_fitness,
                        self.elite.nodes_count,
                        float(gen_diversity),
                        np.std(population.fit),
                        " ".join(
                            [str(ind.nodes_count) for ind in population.population]
                        ),
                        " ".join([str(f) for f in population.fit]),
                        log,
                    ]

                else:
                    add_info = [self.elite.test_fitness, self.elite.nodes_count, log]

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
                    self.elite.nodes_count,
                )
