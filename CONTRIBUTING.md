# Developer Tutorial

## Extensibility
Given our primary objective of promoting research, as well as the exploration and expansion of the SLIM-GSGP algorithm through the development of a library focused on simplicity and extensibility, this appendix is intended for researchers interested in extending the `slim_gsgp` library.

The `slim_gsgp` source code is organised into six core modules: `algorithms`, `datasets`, `evaluators`, `initializers`, `selection`, and `utils`.

The following subsections offer developers guidance on the intricacies of the Slim library’s code and modules, highlighting straightforward yet essential considerations to keep in mind when expanding the library. 

### Config files
In addition to the `slim_gsgp` six core modules, the `config` module contains the configuration files for defining all the hyperparameters used in the algorithms provided in the library.

Although hyperparameters can be directly set when using the algorithms, it is highly recommended to configure them through the configuration files to ensure consistency throughout the library.

Therefore, whenever a new functionality is added to any of the six core modules, the configuration files must be updated accordingly.

### Unit Tests
The `slim_gsgp` repository also contains a `test` folder with unit tests implemented through the `Pytest` library. These tests are used to verify that the algorithms and many of its underlying parts are functioning correctly. Developers interested in modifying `slim_gsgp` are advised to regularly run these tests (and add new ones accordingly). Of particular note is the `test_X_immutability` tests. These execute a pre-configured run of the algorithms to ensure that any changes to the general code do not modify the results. In certain scenarios, a developer might intend to perform these modifications, in this case we advise the usage of the `pytest.mark.skip` decorator to disable that specific test.

## Adding a new evaluator
**Script path:** `slim_gsgp/evaluators/fitness_functions`

The fitness functions must accept two one-dimensional `PyTorch` Tensor (torch.Tensor) objects as input: one representing the target values and the other representing the individual's predicted values. The output of the newly implemented evaluator should be a one-dimensional `PyTorch` Tensor containing the individual's fitness value.

## Adding a new initializer
**Script path:** `slim_gsgp/initializers/initializers`

The initialization functions must accept six different inputs:
- The population size
- The maximum initial depth allowed
- The set of functions
- The set of terminals
- The set of constants
- The probability of selecting a constant over a terminal

The output of the initialization functions should be a `list` of tree-like structures represented as tuples.

If additional parameters are needed, it is recommended to use a nested function. In this approach, the outer function takes all the required specific parameters and returns an inner function, which receives the parameters common to all initializers.

The sets of function, terminals, and constants should be `Python dictionaries`.

The functions `dictionary` must have the name of the function as its `key` and another `dictionary` as its `value`. This inner `dictionary` must have two elements: (i) the `function` `key` whose `value` is the function to be applied to the inputs; and (ii) the `arity` `key` whose `value` is the function arity.

The terminals and constants `dictionaries` must have the name of the corresponding terminal or constant as its `key` and the value of the corresponding terminal or constant as its `value`.

## Adding a new selection algorithm
**Script path:** `slim_gsgp/selection/selection_algorithms`

Each selection algorithm needs to take as input a `Population` object and return a single individual of this `Population`.

If additional parameters are needed, it is recommended to use a nested function. In this approach, the outer function takes all the required specific parameters and returns an inner function, which receives the parameters common to all selectors.

The selection algorithm should also account for whether the optimization problem is one of minimization or maximization, ensuring that the appropriate criteria are used to select individuals.

Following the approach adopted in the provided tournament selection, we recommend a double implementation of the same selection algorithm: one function for maximization and another for minimization problems.

## Adding a New Genetic Operator
When implementing a new genetic operator, it is crucial to differentiate among the three existing algorithms: Genetic Programming (GP), Geometric Semantic Genetic Programming (GSGP) and Semantic Learning algorithm based on Inflate and deflate Mutation (SLIM).

The newly implemented operator must be placed in the corresponding algorithm’s folder within the `slim_gsgp/algorithms/` directory, specifically under the `operators` sub-directory.

### GP
In GP, operators take as input the *structure* of one or two trees, depending on whether the operator is a mutation or crossover. They return the *structure* of the newly modified tree(s). The evaluation of the new tree(s) is performed outside of the operator function.

### Geometric Semantic GP
In GSGP, operators can take as input the *individual(s) object* or the *semantics* of the individual(s), along with other necessary inputs for mutation or crossover (e.g., mutation step, random tree(s), etc.). Independently of the input used, the GSGP operators must return the semantics of the offspring.

GSGP genetic operators must have the `new_data` parameter.

It is used to indicate whether the semantics of the new tree needs to be recalculated from scratch on new data, or if the individual semantics can be used.

It is also crucial to ensure that the **reconstruct** option of GSGP individuals is `True` during the evolutionary process; otherwise, final tree predictions on new data cannot be performed.

### SLIM-GSGP
In SLIM-GSGP, the genetic operators take an `Individual` object as input and either add or remove a block, depending on whether the operation is an inflate or deflate mutation.

This modification affects both the linked list of the individual’s structure (`individual.collection`) and its semantics list (`individual.train_semantics`).

Therefore, the mutate `Individual` object with its updated structure and semantics should be returned.

## Adding a new dataset
The datasets used in the algorithm are always `PyTorch Tensor` objects and are divided into input and target. To make use of a dataset that is not one of the many provided in the `slim_gsgp` library, the developer has several options:

1. The developer can place the dataset in the `slim_gsgp/datasets/data` folder and create a corresponding loading function in the `data_loader.py` file. This function should follow the same format as the existing ones, accepting a boolean parameter `X_y` to specify whether the data should be returned as two separate tensors (for the data `X` and the target `y`). Once implemented, this function can then be called in the main files, as demonstrated in the `slim_gsgp/examples` files.

2. Similar to the user, the developer can utilize the `load_pandas_df` function from `slim_gsgp/datasets/data/data_loader.py` within their main file. To do this, the user must load their data into a pandas dataframe, ensuring that the target variable is the last column, and pass the dataframe to the `load_pandas_df` function with `X_y = True` to receive two `torch.Tensors` corresponding to the input features and target data.

3. The developer can also load their data in any format and manually convert it into a `torch.Tensor`, splitting it into `X` and `y`. For this conversion, the [torch.as_tensor](https://pytorch.org/docs/stable/generated/torch.as_tensor.html) function can be used.
