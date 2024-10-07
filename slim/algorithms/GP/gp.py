"""
Genetic Programming (GP) module.
"""

import random
import time

import numpy as np
import torch
from slim.algorithms.GP.representations.population import Population
from slim.algorithms.GP.representations.tree import Tree
from slim.utils.diversity import niche_entropy
from slim.utils.logger import logger
from slim.utils.utils import verbose_reporter


class GP:
    def __init__(
        self,
        pi_init,
        initializer,
        selector,
        mutator,
        crossover,
        find_elit_func,
        p_m=0.2,
        p_xo=0.8,
        pop_size=100,
        seed=0,
        settings_dict=None,
    ):
        """
        Initialize the Genetic Programming algorithm.

        Parameters
        ----------
        pi_init : dict
            Dictionary with all the parameters needed for evaluation.
        initializer : Callable
            Function to initialize the population.
        selector : Callable
            Function to select individuals for crossover/mutation.
        mutator : Callable
            Function to mutate individuals.
        crossover : Callable
            Function to perform crossover between individuals.
        find_elit_func : Callable
            Function to find elite individuals.
        p_m : float, optional
            Probability of mutation. Default is 0.2.
        p_xo : float, optional
            Probability of crossover. Default is 0.8.
        pop_size : int, optional
            Size of the population. Default is 100.
        seed : int, optional
            Seed for random number generation. Default is 0.
        settings_dict : dict, optional
            Additional settings dictionary.
        """
        self.pi_init = pi_init
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.mutator = mutator
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed
        self.find_elit_func = find_elit_func
        self.settings_dict = settings_dict

        Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        Tree.TERMINALS = pi_init["TERMINALS"]
        Tree.CONSTANTS = pi_init["CONSTANTS"]

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
        max_depth=None,
        ffunction=None,
        n_elites=1,
        tree_pruner=None,
        depth_calculator=None,
        n_jobs = 1
    ):
        """
        Execute the Genetic Programming algorithm.


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
        n_iter : int, optional
            Number of iterations. Default is 20.
        elitism : bool, optional
            Whether to use elitism. Default is True.
        log : int, optional
            Logging level. Default is 0.
        verbose : int, optional
            Verbosity level. Default is 0.
        test_elite : bool, optional
            Whether to test elite individuals. Default is False.
        log_path : str, optional
            Path to save logs. Default is None.
        run_info : list, optional
            Information about the current run. Default is None.
        max_depth : int, optional
            Maximum depth of the tree. Default is None.
        ffunction : function, optional
            Fitness function. Default is None.
        n_elites : int, optional
            Number of elites. Default is 1.
        tree_pruner : function, optional
            Function to prune trees. Default is None.
        depth_calculator : function, optional
            Function to calculate tree depth. Default is None.
        n_jobs : int, optional
            The number of jobs for parallel processing. Default is 1.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        # Initialize the population
        population = Population(
            [Tree(tree) for tree in self.initializer(**self.pi_init)]
        )
        population.evaluate(ffunction, X=X_train, y=y_train, n_jobs=n_jobs)

        end = time.time()
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        if test_elite:
            self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

        if log != 0:
            self.log_initial_population(
                population, end - start, log, log_path, run_info
            )

        if verbose != 0:
            verbose_reporter(
                curr_dataset.split("load_")[-1],
                0,
                self.elite.fitness,
                self.elite.test_fitness,
                end - start,
                self.elite.node_count,
            )

        for it in range(1, n_iter + 1):
            offs_pop, start = self.evolve_population(
                population,
                ffunction,
                max_depth,
                depth_calculator,
                elitism,
                X_train,
                y_train,
                n_jobs=n_jobs
            )
            population = offs_pop
            end = time.time()

            self.elites, self.elite = self.find_elit_func(population, n_elites)

            if test_elite:
                self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

            if log != 0:
                self.log_generation(
                    it, population, end - start, log, log_path, run_info
                )

            if verbose != 0:
                verbose_reporter(
                    run_info[-1],
                    it,
                    self.elite.fitness,
                    self.elite.test_fitness,
                    end - start,
                    self.elite.node_count,
                )

    def evolve_population(
        self,
        population,
        ffunction,
        max_depth,
        depth_calculator,
        elitism,
        X_train,
        y_train,
        n_jobs=1
    ):
        """
        Evolve the population for one generation.

        Args:
            population (Population): Current population.
            ffunction (function): Fitness function.
            max_depth (int): Maximum depth of the tree.
            depth_calculator (function): Function to calculate tree depth.
            elitism (bool): Whether to use elitism.
            X_train (torch.Tensor): Training data features.
            y_train (torch.Tensor): Training data labels.
            n_jobs (int): The number of jobs for the joblib library Parallel parallelization.

        Returns:
            Population: Evolved population.
            float: Start time of evolution.
        """
        offs_pop = []
        start = time.time()

        if elitism:
            offs_pop.extend(self.elites)

        while len(offs_pop) < self.pop_size:
            if random.random() < self.p_xo:
                p1, p2 = self.selector(population), self.selector(population)
                while p1 == p2:
                    p1, p2 = self.selector(population), self.selector(population)

                offs1, offs2 = self.crossover(
                    p1.repr_,
                    p2.repr_,
                    tree1_n_nodes=p1.node_count,
                    tree2_n_nodes=p2.node_count,
                )

                if max_depth is not None:
                    while (
                        depth_calculator(offs1) > max_depth
                        or depth_calculator(offs2) > max_depth
                    ):
                        offs1, offs2 = self.crossover(
                            p1.repr_,
                            p2.repr_,
                            tree1_n_nodes=p1.node_count,
                            tree2_n_nodes=p2.node_count,
                        )
                else:
                    offs1, offs2 = self.crossover(
                        p1.repr_,
                        p2.repr_,
                        tree1_n_nodes=p1.node_count,
                        tree2_n_nodes=p2.node_count,
                    )

                offspring = [offs1, offs2]
            else:
                p1 = self.selector(population)
                offs1 = self.mutator(p1.repr_, num_of_nodes=p1.node_count)

                if max_depth is not None:
                    while depth_calculator(offs1) > max_depth:
                        offs1 = self.mutator(p1.repr_, num_of_nodes=p1.node_count)
                else:
                    offs1 = self.mutator(p1.repr_, num_of_nodes=p1.node_count)

                offspring = [offs1]

            offs_pop.extend([Tree(child) for child in offspring])

        if len(offs_pop) > population.size:
            offs_pop = offs_pop[: population.size]

        offs_pop = Population(offs_pop)
        offs_pop.evaluate(ffunction, X=X_train, y=y_train, n_jobs=n_jobs)
        return offs_pop, start

    def log_initial_population(self, population, elapsed_time, log, log_path, run_info):
        """
        Log the initial population.

        Args:
            population (Population): The population to log.
            elapsed_time (float): Time taken for the process.
            log (int): Logging level.
            log_path (str): Path to save logs.
            run_info (list): Information about the current run.

        Returns:
            None
        """
        if log == 2:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                log,
            ]
        elif log == 3:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        elif log == 4:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        else:
            add_info = [self.elite.test_fitness, self.elite.node_count, log]

        logger(
            log_path,
            0,
            self.elite.fitness,
            elapsed_time,
            float(population.nodes_count),
            additional_infos=add_info,
            run_info=run_info,
            seed=self.seed,
        )

    def log_generation(
        self, generation, population, elapsed_time, log, log_path, run_info
    ):
        """
        Log the results for the current generation.

        Args:
            generation (int): Current generation number.
            population (Population): Current population.
            elapsed_time (float): Time taken for the process.
            log (int): Logging level.
            log_path (str): Path to save logs.
            run_info (list): Information about the current run.

        Returns:
            None
        """
        if log == 2:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                log,
            ]
        elif log == 3:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        elif log == 4:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        else:
            add_info = [self.elite.test_fitness, self.elite.node_count, log]

        logger(
            log_path,
            generation,
            self.elite.fitness,
            elapsed_time,
            float(population.nodes_count),
            additional_infos=add_info,
            run_info=run_info,
            seed=self.seed,
        )
