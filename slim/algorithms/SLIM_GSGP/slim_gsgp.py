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

        Parameters
        ----------
        pi_init : dict
            Dictionary with all the parameters needed for candidate solutions initialization.
        initializer : Callable
            Function to initialize the population.
        selector : Callable
            Function to select individuals.
        inflate_mutator : Callable
            Function for inflate mutation.
        deflate_mutator : Callable
            Function for deflate mutation.
        ms : Callable
            Mutation step function.
        crossover : Callable
            Crossover function.
        find_elit_func : Callable
            Function to find elite individuals.
        p_m : float
            Probability of mutation. Default is 1.
        p_xo : float
            Probability of crossover. Default is 0.
        p_inflate : float
            Probability of inflate mutation. Default is 0.3.
        p_deflate : float
            Probability of deflate mutation. Default is 0.7.
        pop_size : int
            Size of the population. Default is 100.
        seed : int
            Random seed for reproducibility. Default is 0.
        operator : {'sum', 'prod'}
            Operator to apply to the semantics, either "sum" or "prod". Default is "sum".
        copy_parent : bool
            Whether to copy the parent when mutation is not possible. Default is True.
        two_trees : bool
            Indicates if two trees are used. Default is True.
        settings_dict : dict
            Additional settings passed as a dictionary.

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
        self.two_trees = two_trees
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
        n_jobs=1):
        """
        Solve the optimization problem using SLIM_GSGP.

        Parameters
        ----------
        X_train : array-like
            Training input data.
        X_test : array-like
            Testing input data.
        y_train : array-like
            Training output data.
        y_test : array-like
            Testing output data.
        curr_dataset : str or int
            Identifier for the current dataset.
        run_info : dict
            Information about the current run.
        n_iter : int
            Number of iterations. Default is 20.
        elitism : bool
            Whether elitism is used during evolution. Default is True.
        log : int or str
            Logging level (e.g., 0 for no logging, 1 for basic, etc.). Default is 0.
        verbose : int
            Verbosity level for logging outputs. Default is 0.
        test_elite : bool
            Whether elite individuals should be tested. Default is False.
        log_path : str
            File path for saving log outputs. Default is None.
        ffunction : function
            Fitness function used to evaluate individuals. Default is None.
        max_depth : int
            Maximum depth for the trees. Default is 17.
        n_elites : int
            Number of elite individuals to retain during selection. Default is True.
        reconstruct : bool
            Indicates if reconstruction of the solution is needed. Default is True.
        n_jobs : int
            Maximum number of concurrently running jobs for joblib parallelization. Default is 1.

        """

        if test_elite and (X_test is None or y_test is None):
            raise Exception('If test_elite is True you need to provide a test dataset')

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # starting time count
        start = time.time()

        # creating the initial population
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

        # calculating initial population semantics
        population.calculate_semantics(X_train)

        # evaluating the initial population
        population.evaluate(ffunction, y=y_train, operator=self.operator, n_jobs=n_jobs)

        end = time.time()

        # setting up the elite(s)
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        # calculating the testing semantics and the elite's testing fitness if test_elite is true
        if test_elite:
            population.calculate_semantics(X_test, testing=True)
            self.elite.evaluate(
                ffunction, y=y_test, testing=True, operator=self.operator
            )

        # logging the results based on the log level
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

        # displaying the results on console if verbose level is more than 0
        if verbose != 0:
            verbose_reporter(
                curr_dataset,
                0,
                self.elite.fitness,
                self.elite.test_fitness,
                end - start,
                self.elite.nodes_count,
            )

        # begining the evolution process
        for it in range(1, n_iter + 1, 1):
            # starting an empty offspring population
            offs_pop, start = [], time.time()

            # adding the elite to the offspring population, if applicable
            if elitism:
                offs_pop.extend(self.elites)

            # filling the offspring population
            while len(offs_pop) < self.pop_size:

                # choosing between crossover and mutation

                if random.random() < self.p_xo:

                    p1, p2 = self.selector(population), self.selector(population)
                    while p1 == p2:
                        # choosing parents
                        p1, p2 = self.selector(population), self.selector(population)
                    pass  # future work on slim implementations should invent crossover
                else:
                    # so, mutation was selected. Now deflation or inflation is selected.
                    if random.random() < self.p_deflate:

                        # selecting the parent to deflate
                        p1 = self.selector(population)

                        # if the parent has only one block, it cannot be deflated
                        if p1.size == 1:
                            # if copy parent is set to true, the parent who cannot be deflated will be copied as the offspring
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
                                # if we choose to not copy the parent, we inflate it instead
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
                            # if the size of the parent is more than 1, normal deflation can occur
                            off1 = self.deflate_mutator(p1, reconstruct=reconstruct)

                    # inflation mutation was selected
                    else:

                        # selecting a parent to inflate
                        p1 = self.selector(population)

                        # determining the random mutation step
                        ms_ = self.ms()

                        # if the chosen parent is already at maximum depth and therefore cannot be inflated
                        if max_depth is not None and p1.depth == max_depth:
                            # if copy parent is set to true, the parent who cannot be inflated will be copied as the offspring
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

                            # if copy parent is false, the parent is deflated instead of inflated
                            else:
                                off1 = self.deflate_mutator(p1, reconstruct=reconstruct)

                        # so the chosen individual can be normally inflated
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

                        # if offspring resulting from inflation exceedes the max depth
                        if max_depth is not None and off1.depth > max_depth:
                            # if copy parent is set to true, the offspring is discarded and the parent is chosen instead
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
                                # otherwise, deflate the parent
                                off1 = self.deflate_mutator(p1, reconstruct=reconstruct)

                    # adding the new offspring to the offspring population
                    offs_pop.append(off1)

            # removing any excess individuals from the offspring population
            if len(offs_pop) > population.size:

                offs_pop = offs_pop[: population.size]

            # turning the offspring population into a Population
            offs_pop = Population(offs_pop)
            # calculating the offspring population semantics
            offs_pop.calculate_semantics(X_train)

            # evaluating the offspring population
            offs_pop.evaluate(ffunction, y=y_train, operator=self.operator, n_jobs=n_jobs)

            # replacing the current population with the offspring population P = P'
            population = offs_pop
            self.population = population

            end = time.time()

            # setting the new elite(s)
            self.elites, self.elite = self.find_elit_func(population, n_elites)

            # calculating the testing semantics and the elite's testing fitness if test_elite is true
            if test_elite:
                self.elite.calculate_semantics(X_test, testing=True)
                self.elite.evaluate(
                    ffunction, y=y_test, testing=True, operator=self.operator
                )

            # logging the results based on the log level
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

            # displaying the results on console if verbose level is more than 0
            if verbose != 0:
                verbose_reporter(
                    run_info[-1],
                    it,
                    self.elite.fitness,
                    self.elite.test_fitness,
                    end - start,
                    self.elite.nodes_count,
                )
