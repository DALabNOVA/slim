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
Population Class for SLIM GSGP using PyTorch.
"""
import torch


_EVALUATION_BUFFER_BYTES = 64 * 1024 * 1024


class Population:
    def __init__(self, population):
        """
        Initialize the Population with a list of individuals.

        Parameters
        ----------
        population : list
            The list of individuals in the population.

        Returns
        -------
        None
        """
        self.population = population
        self.size = len(population)
        self.nodes_count = sum([ind.nodes_count for ind in population])
        self.fit = None
        self.train_semantics = None
        self.test_semantics = None

    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for each individual in the population.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data for calculating semantics.
        testing : bool, optional
            Boolean indicating if the calculation is for testing semantics.

        Returns
        -------
        None
        """
        semantics_attribute = "test_semantics" if testing else "train_semantics"
        for individual in self.population:
            individual.calculate_semantics(inputs, testing)
        setattr(
            self,
            semantics_attribute,
            [getattr(individual, semantics_attribute) for individual in self.population],
        )

    def __len__(self):
        """
        Return the size of the population.

        Returns
        -------
        int
            Size of the population.
        """
        return self.size

    def __getitem__(self, item):
        """
        Get an individual from the population by index.

        Parameters
        ----------
        item : int
            Index of the individual to retrieve.

        Returns
        -------
        Individual
            The individual at the specified index.
        """
        return self.population[item]

    def evaluate_no_parall(self, ffunction, y, operator="sum"):
        """
        Evaluate the population using a fitness function (without parallelization).
        This function is not currently in use, but has been retained for potential future use
        at the developer's discretion.

        Parameters
        ----------
        ffunction : Callable
            Fitness function to evaluate the individuals.
        y : torch.Tensor
            Expected output (target) values.
        operator : str, optional
            Operator to apply to the semantics. Default is "sum".

        Returns
        -------
        None
        """
        # evaluating all the individuals in the population
        [
            individual.evaluate(ffunction, y, operator=operator)
            for individual in self.population
        ]
        # defining the fitness of the population to be a list with the fitnesses of all individuals in the population
        self.fit = [individual.fitness for individual in self.population]

    def evaluate(self, ffunction, y, operator="sum"):
        """
        Evaluate the population using a fitness function.

        Parameters
        ----------
        ffunction : Callable
            Fitness function to evaluate the individuals.
        y : torch.Tensor
            Expected output (target) values.
        operator : str, optional
            Operator to apply to the semantics ("sum" or "mul"). Default is "sum".
        Returns
        -------
        None
        """
        if operator not in ("sum", "mul"):
            raise ValueError("operator must be 'sum' or 'mul'")
        if not self.population:
            self.fit = []
            return

        semantics = self.population[0].train_semantics
        if semantics.ndim != 2:
            raise ValueError("SLIM train semantics must have shape (blocks, samples)")
        n_samples = semantics.shape[1]
        row_bytes = n_samples * semantics.element_size()
        chunk_size = max(1, min(self.size, _EVALUATION_BUFFER_BYTES // max(1, row_bytes)))

        with torch.no_grad():
            workspace = torch.empty(
                (chunk_size, n_samples), dtype=semantics.dtype, device=semantics.device
            )
            fit_chunks = []
            for start in range(0, self.size, chunk_size):
                stop = min(start + chunk_size, self.size)
                predictions = workspace[: stop - start]
                for row, individual in enumerate(self.population[start:stop]):
                    if individual.train_semantics.shape[0] == 1:
                        predictions[row].copy_(individual.train_semantics[0])
                    elif operator == "sum":
                        torch.sum(individual.train_semantics, dim=0, out=predictions[row])
                    else:
                        torch.prod(individual.train_semantics, dim=0, out=predictions[row])

                predictions.clamp_(-1_000_000_000_000.0, 1_000_000_000_000.0)
                scores = ffunction(y, predictions)
                if scores.shape != (stop - start,):
                    raise ValueError(
                        "SLIM fitness functions must return one scalar per individual; "
                        f"received shape {tuple(scores.shape)}"
                    )
                if not torch.isfinite(scores).all():
                    raise ValueError("SLIM fitness functions must return finite values")
                fit_chunks.append(scores)

            fits = torch.cat(fit_chunks)
            self.fit = fits.detach().cpu().tolist()

        # Assign individuals' fitness as an attribute
        for ind, f in zip(self.population, self.fit):
            ind.fitness = f

