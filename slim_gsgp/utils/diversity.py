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
import torch
from scipy.stats import entropy


def niche_entropy(repr_, n_niches=10):
    """
    Calculate the niche entropy of a population.

    Parameters
    ----------
    repr_ : list
        The list of individuals in the population.
    n_niches : int
        Number of niches to divide the population into.

    Returns
    -------
    float
        The entropy of the distribution of individuals across niches.
    Notes
    -----
    https://www.semanticscholar.org/paper/Entropy-Driven-Adaptive-RoscaComputer/ab5c8a8f415f79c5ec6ff6281ed7113736615682
    https://strathprints.strath.ac.uk/76488/1/Marchetti_etal_Springer_2021_Inclusive_genetic_programming.pdf
    """

    num_nodes = [len(ind) - 1 for ind in repr_]
    min_ = min(num_nodes)
    max_ = max(num_nodes)
    pop_size = len(repr_)
    stride = (max_ - min_) / n_niches

    distributions = []
    for i in range(1, n_niches + 1):
        distribution = (
            sum((i - 1) * stride + min_ <= x < i * stride + min_ for x in num_nodes)
            / pop_size
        )
        if distribution > 0:
            distributions.append(distribution)

    return entropy(distributions)


def gsgp_pop_div_from_vectors(sem_vectors):
    """
    Calculate the diversity of a population from semantic vectors.

    Parameters
    ----------
    sem_vectors : torch.Tensor
        The tensor of semantic vectors.

    Returns
    -------
    float
        The average pairwise distance between semantic vectors.
    Notes
    -----
    https://ieeexplore.ieee.org/document/9283096
    """
    return torch.sum(torch.cdist(sem_vectors, sem_vectors)) / (
        sem_vectors.shape[0] ** 2
    )
