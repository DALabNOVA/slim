import torch
from scipy.stats import entropy


def niche_entropy(repr_, n_niches=10):
    """
    Calculate the niche entropy of a population.

    Args:
        repr_ (list): List of individuals in the population.
        n_niches (int): Number of niches to divide the population into.

    Returns:
        float: The entropy of the distribution of individuals across niches.
    """
    # https://www.semanticscholar.org/paper/Entropy-Driven-Adaptive-RoscaComputer/ab5c8a8f415f79c5ec6ff6281ed7113736615682
    # https://strathprints.strath.ac.uk/76488/1/Marchetti_etal_Springer_2021_Inclusive_genetic_programming.pdf

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

    Args:
        sem_vectors (torch.Tensor): Tensor of semantic vectors.

    Returns:
        float: The average pairwise distance between semantic vectors.
    """
    # https://ieeexplore.ieee.org/document/9283096
    return torch.sum(torch.cdist(sem_vectors, sem_vectors)) / (
        sem_vectors.shape[0] ** 2
    )
