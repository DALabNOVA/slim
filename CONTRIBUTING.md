### Extensibility
#### Introduction

Given our primary objective of promoting research, as well as the exploration and expansion of the SLIM-GSGP algorithm through the development of a library focused on simplicity and extensibility, this section is intended for researchers interested in extending the `slim_gsgp` library.

The `slim_gsgp` source code is organized into six core modules: `algorithms`, `datasets`, `evaluators`, `initializers`, `selection`, and `utils`. The following subsections provide guidance for developers on expanding the Slim library.

#### Config Files

In addition to the six core modules, the `config` module contains configuration files for defining all the hyperparameters used in the algorithms provided in the library. While hyperparameters can be set directly, it is recommended to configure them through the configuration files to maintain consistency. When new functionality is added, update the configuration files accordingly.

#### Unit Tests

The `slim_gsgp` repository contains a `test` folder with unit tests implemented using the `Pytest` library. These tests verify the correct functioning of algorithms. Developers modifying the library should run these tests regularly and add new ones as needed. Of particular note is the `test_X_immutability` tests, which ensure that algorithm changes do not modify the results. If changes are intended, use the `pytest.mark.skip` decorator to disable specific tests.

### Adding a New Evaluator
#### Script path: `slim_gsgp/evaluators/fitness_functions`

The fitness functions must accept two one-dimensional `PyTorch` Tensor (torch.Tensor) objects as input: one representing the target values and the other representing the individual's predicted values. The output of the newly implemented evaluator should be a one-dimensional `PyTorch` Tensor containing the individual's fitness value.

### Adding a New Initializer
#### Script path: `slim_gsgp/initializers/initializers`

The initialization functions must accept six inputs:

- Population size
- Maximum initial depth allowed
- Set of functions
- Set of terminals
- Set of constants
- Probability of selecting a constant over a terminal

The output should be a `list` of tree-like structures represented as tuples. If additional parameters are needed, use a nested function. The sets of functions, terminals, and constants should be `Python dictionaries`. The functions dictionary must contain the name of the function as the `key` and a nested dictionary with two elements: `function` (the function to apply) and `arity` (the number of inputs).

### Adding a New Selection Algorithm
#### Script path: `slim_gsgp/selection/selection_algorithms`

Each selection algorithm must accept a `Population` object and return an individual from this `Population`. If additional parameters are required, use a nested function. The algorithm should also account for minimization or maximization optimization problems, and we recommend a double implementation for both types.

### Adding a New Genetic Operator
#### GP
In Genetic Programming (GP), operators take as input the structure of one or two trees (depending on whether the operator is mutation or crossover) and return the structure of the newly modified tree(s). The evaluation is done outside the operator function.

#### Geometric Semantic GP (GSGP)
GSGP operators can take as input individual(s) objects or their semantics, and return the semantics of the offspring. GSGP operators must also account for the `new_data` parameter, ensuring that semantics are recalculated when required.

#### SLIM-GSGP
SLIM-GSGP operators modify both the individual's structure (`individual.collection`) and semantics (`individual.train_semantics`) when performing inflate or deflate mutations. The modified `Individual` object must be returned.

### Adding a New Dataset
The datasets used in the algorithm are `PyTorch Tensor` objects, divided into input and target. To use a new dataset, the developer has the following options:

1. Place the dataset in the `slim_gsgp/datasets/data` folder and create a corresponding loading function in `data_loader.py`.
2. Use the `load_pandas_df` function from `slim_gsgp/datasets/data/data_loader.py`.
3. Manually convert the data into `torch.Tensor` and split it into `X` and `y`.
